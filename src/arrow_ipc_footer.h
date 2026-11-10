#pragma once

#include <arrow/ipc/api.h>
#include <vector>

// Arrow C++ 26 exports this function, but does not install its declaration.
// Keep the private ABI in one place and check the symbol when configuring.
namespace arrow::ipc::internal
{
struct FileBlock
{
  int64_t offset;
  int32_t metadata_length;
  int64_t body_length;
};

ARROW_EXPORT Status WriteFileFooter(const Schema &schema, const std::vector<FileBlock> &dictionaries,
                                    const std::vector<FileBlock> &record_batches,
                                    const std::shared_ptr<const KeyValueMetadata> &metadata,
                                    io::OutputStream *out);
} // namespace arrow::ipc::internal
