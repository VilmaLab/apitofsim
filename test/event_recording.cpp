#include <doctest/doctest.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <nanoarrow/nanoarrow_ipc.hpp>
#include "event_recording.h"
#include "parquet_event_writer.h"

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

TEST_CASE("Native Parquet rejects malformed batches and propagates file errors")
{
  ArchiveFixture fixture;
  CHECK_THROWS_AS(ParquetEventWriter(fixture.path.string(), "archive", 1, false, 0), std::invalid_argument);
  {
    ParquetEventWriter writer((fixture.directory / "missing" / "events.parquet").string(), "archive", 1, true, 4);
    CHECK_THROWS_AS(writer.close(), std::runtime_error);
  }
  for (const int workers : {1, 4})
  {
    ParquetEventWriter writer(fixture.path.string(), "archive", 1, true, workers);
    auto data = std::make_shared<EventColumns>(columns(2));
    data->event_indices = {0, 1};
    data->result_ids = {1};
    writer.write(data);
    CHECK_THROWS_AS(writer.close(), std::invalid_argument);
  }
}

TEST_CASE("Native Parquet drains queued batches across row groups and closes empty files")
{
  ArchiveFixture fixture;
  for (const bool zstd : {false, true})
    for (const int workers : {1, 4})
    {
      ParquetEventWriter writer(fixture.path.string(), "archive", 1, zstd, workers);
      for (const size_t size : {size_t{65530}, size_t{20}, size_t{65540}})
      {
        auto data = std::make_shared<EventColumns>(columns(size));
        data->event_indices.assign(size, 0);
        data->result_ids.assign(size, 1);
        writer.write(data); // The queue owns each batch after this scope.
      }
      writer.close();
      writer.close();
      const auto bytes = contents(fixture.path);
      CHECK(bytes.substr(0, 4) == "PAR1");
      CHECK(bytes.substr(bytes.size() - 4) == "PAR1");
    }
  ParquetEventWriter empty(fixture.path.string(), "archive", 1, false, 1);
  empty.close();
  const auto bytes = contents(fixture.path);
  CHECK(bytes.substr(0, 4) == "PAR1");
  CHECK(bytes.substr(bytes.size() - 4) == "PAR1");
}

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
  // Both batches and the original float data remain readable by the independent
  // nanoarrow decoder. Python tests additionally use PyArrow's file reader.
  nanoarrow::ipc::UniqueDecoder decoder;
  REQUIRE(ArrowIpcDecoderInit(decoder.get()) == NANOARROW_OK);
  ArrowError error{};
  ArrowBufferView input{};
  input.data.data = appended.data();
  input.size_bytes = static_cast<int64_t>(appended.size());
  REQUIRE(ArrowIpcDecoderVerifyFooter(decoder.get(), input, &error) == NANOARROW_OK);
  REQUIRE(ArrowIpcDecoderDecodeFooter(decoder.get(), input, &error) == NANOARROW_OK);
  REQUIRE(decoder->footer->schema.n_children == 22);
  REQUIRE(decoder->footer->record_batch_blocks.size_bytes == 2 * sizeof(ArrowIpcFileBlock));
  const auto *blocks = reinterpret_cast<const ArrowIpcFileBlock *>(decoder->footer->record_batch_blocks.data);
  const auto first_end = static_cast<size_t>(blocks[0].offset + blocks[0].metadata_length + blocks[0].body_length);
  CHECK(original.substr(0, first_end) == appended.substr(0, first_end));
  double value = 0;
  std::memcpy(&value, appended.data() + blocks[0].offset + blocks[0].metadata_length, sizeof(value));
  CHECK(value == 1.5);
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

TEST_CASE("Native IPC rejects damaged framing before overwriting the archive")
{
  ArchiveFixture fixture;
  IpcEventWriter writer(fixture.path.string(), false, "archive");
  writer.close();
  const auto original = contents(fixture.path);
  uint32_t footer_size = 0;
  for (unsigned int i = 0; i < 4; ++i)
    footer_size |= static_cast<uint32_t>(static_cast<unsigned char>(original[original.size() - 10 + i])) << (8U * i);
  REQUIRE(footer_size < original.size() - 18);
  for (const size_t offset : {size_t{0}, original.size() - footer_size - 18})
  {
    auto damaged = original;
    damaged[offset] = 'X';
    {
      std::ofstream output(fixture.path, std::ios::binary | std::ios::trunc);
      output.write(damaged.data(), static_cast<std::streamsize>(damaged.size()));
    }
    CHECK_THROWS_AS(IpcEventWriter(fixture.path.string(), true, "archive"), std::runtime_error);
    CHECK(contents(fixture.path) == damaged);
  }
}
