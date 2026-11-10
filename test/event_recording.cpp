#include <doctest/doctest.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <arrow/array.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/record_batch.h>
#include "event_recording.h"

namespace
{
struct ArchiveFixture
{
  std::filesystem::path directory;
  std::filesystem::path path;
  ArchiveFixture()
  {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    directory = std::filesystem::temp_directory_path() / ("apitofsim-ipc-" + std::to_string(stamp) + "-" + std::to_string(std::random_device{}()));
    REQUIRE(std::filesystem::create_directory(directory));
    path = directory / "events.arrow";
  }
  ~ArchiveFixture()
  {
    std::filesystem::remove_all(directory);
  }
};

std::string contents(const std::filesystem::path &path)
{
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

EventColumns columns(size_t n)
{
  EventColumns out;
  for (auto &values : out.values)
    values.assign(n, 1.5);
  for (auto &indices : out.indices)
    indices.assign(n, 0);
  return out;
}
} // namespace

TEST_CASE("Native IPC appends batches without rewriting committed event payloads")
{
  ArchiveFixture fixture;
  auto data = columns(3);
  {
    IpcEventWriter writer(fixture.path.string(), false, "archive");
    writer.write(data, 7, {0, 1, 2});
    CHECK(writer.rows_written() == 3);
    CHECK(writer.batch_count() == 1);
    data.clear(3); // The synchronous encoder must not retain borrowed vectors.
    writer.set_metadata(R"({"7":{"status":"complete"}})");
    writer.close();
    writer.close();
    CHECK_THROWS_AS(writer.write(columns(1), 8, {0}), std::runtime_error);
  }
  auto original = contents(fixture.path);
  REQUIRE(original.substr(0, 6) == "ARROW1");
  {
    IpcEventWriter writer(fixture.path.string(), true, "archive");
    writer.write(columns(2), 8, {0, 1});
    CHECK(writer.rows_written() == 2);
    CHECK(writer.batch_count() == 2);
    writer.close();
  }
  const auto appended = contents(fixture.path);
  REQUIRE(appended.size() > original.size());
  // All bytes before the original EOS marker must survive append.
  const auto trailer = original.size() - 10;
  uint32_t footer_size = 0;
  for (unsigned int i = 0; i < 4; ++i)
    footer_size |= static_cast<uint32_t>(static_cast<unsigned char>(original[trailer + i])) << (8U * i);
  REQUIRE(footer_size < trailer - 8);
  const auto first_end = trailer - footer_size - 8;
  CHECK(original.substr(0, first_end) == appended.substr(0, first_end));

  auto input = std::make_shared<arrow::io::BufferReader>(arrow::Buffer::Wrap(appended.data(), appended.size()));
  auto reader = arrow::ipc::RecordBatchFileReader::Open(input);
  REQUIRE(reader.ok());
  CHECK((*reader)->schema()->num_fields() == 22);
  REQUIRE((*reader)->num_record_batches() == 2);
  auto first = (*reader)->ReadRecordBatch(0);
  REQUIRE(first.ok());
  CHECK(dynamic_cast<const arrow::DoubleArray &>(*(*first)->column(0)).Value(0) == 1.5);
  CHECK_THROWS_AS(IpcEventWriter(fixture.path.string(), true, "wrong-archive"), std::runtime_error);
  CHECK(contents(fixture.path) == appended);
}

TEST_CASE("Native IPC rejects inconsistent columns and supports empty files")
{
  ArchiveFixture fixture;
  IpcEventWriter writer(fixture.path.string(), false, "empty");
  auto data = columns(2);
  data.values[1].clear();
  CHECK_THROWS_AS(writer.write(data, 1, {0, 1}), std::invalid_argument);
  writer.close();
  IpcEventWriter reopened(fixture.path.string(), true, "empty");
  CHECK(reopened.batch_count() == 0);
  reopened.close();
}
