////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTME_CORE_INC_DRIVER_H_
#define HSA_RUNTME_CORE_INC_DRIVER_H_

#include <limits>
#include <string>
#include <sys/mman.h>

#include "core/inc/memory_region.h"
#include "inc/hsa.h"

namespace rocr {
namespace core {

class Queue;

struct DriverVersionInfo {
  uint32_t major;
  uint32_t minor;
};

enum class DriverQuery { GET_DRIVER_VERSION };

enum class DriverType { XDNA = 0, KFD, NUM_DRIVER_TYPES };

/// @brief Handle for exported / imported memory.
struct ShareableHandle {
  uint64_t handle{};

  bool IsValid() const { return handle != 0; }
};

/// @brief Kernel driver interface.
///
/// @details A class used to provide an interface between the core runtime
/// and agent kernel drivers. It also maintains state associated with active
/// kernel drivers.
class Driver {
public:
  /// @brief Converts @ref hsa_access_permission_t to mmap memory protection
  ///        flags.
  __forceinline static int
  PermissionsToMmapFlags(hsa_access_permission_t perms) {
    switch (perms) {
    case HSA_ACCESS_PERMISSION_RO:
      return PROT_READ;
    case HSA_ACCESS_PERMISSION_WO:
      return PROT_WRITE;
    case HSA_ACCESS_PERMISSION_RW:
      return PROT_READ | PROT_WRITE;
    case HSA_ACCESS_PERMISSION_NONE:
    default:
      return PROT_NONE;
    }
  }

  Driver(DriverType kernel_driver_type, std::string devnode_name);
  virtual ~Driver() = default;

  /// @brief Initialize the driver's state after opening.
  virtual hsa_status_t Init() = 0;

  /// @brief Query the kernel-model driver.
  /// @retval HSA_STATUS_SUCCESS if the kernel-model driver query was
  /// successful.
  virtual hsa_status_t QueryKernelModeDriver(DriverQuery query) = 0;

  /// @brief Open a connection to the driver using name_.
  /// @retval HSA_STATUS_SUCCESS if the driver was opened successfully.
  hsa_status_t Open();

  /// @brief Close a connection to the open driver using fd_.
  /// @retval HSA_STATUS_SUCCESS if the driver was opened successfully.
  hsa_status_t Close();

  /// @brief Get driver version information.
  /// @retval DriverVersionInfo containing the driver's version information.
  const DriverVersionInfo &Version() const { return version_; }

  /// @brief Get the properties of a specific agent and initialize the agent
  /// object.
  /// @param agent Agent whose properties we're getting.
  /// @retval HSA_STATUS_SUCCESS if the driver successfully returns the agent's
  ///         properties.
  virtual hsa_status_t GetAgentProperties(Agent &agent) const = 0;

  /// @brief Get the memory properties of a specific node.
  /// @param node_id Node ID of the agent
  /// @param[in, out] mem_region MemoryRegion object whose properties will be
  /// retrieved.
  /// @retval HSA_STATUS_SUCCESS if the driver sucessfully returns the node's
  ///         memory properties.
  virtual hsa_status_t GetMemoryProperties(uint32_t node_id,
                                           MemoryRegion &mem_region) const = 0;

  /// @brief Allocate agent-accessible memory (system or agent-local memory).
  ///
  /// @param[out] mem pointer to newly allocated memory.
  ///
  /// @retval HSA_STATUS_SUCCESS if memory was successfully allocated or
  /// hsa_status_t error code if the memory allocation failed.
  virtual hsa_status_t AllocateMemory(const MemoryRegion &mem_region,
                                      MemoryRegion::AllocateFlags alloc_flags,
                                      void **mem, size_t size,
                                      uint32_t node_id) = 0;

  virtual hsa_status_t FreeMemory(void *mem, size_t size) = 0;

  virtual hsa_status_t CreateQueue(Queue &queue) const = 0;

  virtual hsa_status_t DestroyQueue(Queue &queue) const = 0;

  /// @brief Imports memory using dma-buf.
  ///
  /// @param[in] mem virtual address
  /// @param[in] size memory size in bytes
  /// @param[out] dmabuf_fd dma-buf file descriptor
  /// @param[out] offset memory offset in bytes
  virtual hsa_status_t ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                                    size_t *offset) = 0;

  /// @brief Imports a memory chunk via dma-buf.
  ///
  /// @param[in] dmabuf_fd dma-buf file descriptor
  /// @param[in] agent agent to import the memory for
  /// @param[out] handle handle to the imported memory
  virtual hsa_status_t ImportDMABuf(int dmabuf_fd, core::Agent &agent,
                                    core::ShareableHandle &handle) = 0;

  /// @brief Maps the memory associated with the handle.
  ///
  /// @param[in] handle handle to the memory object
  /// @param[in] mem virtual address associated with the handle
  /// @param[in] offset memory offset in bytes
  /// @param[in] size memory size in bytes
  /// @param[perms] perms new permissions
  virtual hsa_status_t Map(core::ShareableHandle handle, void *mem,
                           size_t offset, size_t size,
                           hsa_access_permission_t perms) = 0;

  /// @brief Unmaps the memory associated with the handle.
  ///
  /// @param[in] handle handle to the memory object
  /// @param[in] mem virtual address associated with the handle
  /// @param[in] offset memory offset in bytes
  /// @param[in] size memory size in bytes
  virtual hsa_status_t Unmap(core::ShareableHandle handle, void *mem,
                             size_t offset, size_t size) = 0;

  /// @brief Releases the object associated with the handle.
  ///
  /// @param[in] handle handle of the object to release
  virtual hsa_status_t
  ReleaseShareableHandle(core::ShareableHandle &handle) = 0;

  /// Unique identifier for supported kernel-mode drivers.
  const DriverType kernel_driver_type_;

protected:
  DriverVersionInfo version_{std::numeric_limits<uint32_t>::max(),
                             std::numeric_limits<uint32_t>::max()};

  const std::string devnode_name_;
  int fd_ = -1;
};

} // namespace core
} // namespace rocr

#endif // header guard
