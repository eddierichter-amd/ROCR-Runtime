////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef HSA_RUNTIME_CORE_INC_AMD_XDNA_DRIVER_H_
#define HSA_RUNTIME_CORE_INC_AMD_XDNA_DRIVER_H_

#include <memory>
#include <unordered_map>
#include <x86intrin.h>

#include "core/inc/driver.h"
#include "core/inc/memory_region.h"
#include "core/driver/xdna/uapi/amdxdna_accel.h"

// Flushes the cache lines assocaited with a BO. This
// is used to sync a BO without going to the xdna driver.
// This is from the XRT KMQ shim.
inline void
clflush_data(const void *base, size_t offset, size_t len)
{
  static long cacheline_size = 0;

  if (!cacheline_size) {
    long sz = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (sz <= 0)
      return;
    cacheline_size = sz;
  }

  const char *cur = (const char *)base;
  cur += offset;
  uintptr_t lastline = (uintptr_t)(cur + len - 1) | (cacheline_size - 1);
  do {
    _mm_clflush(cur);
    cur += cacheline_size;
  } while (cur <= (const char *)lastline);
}

namespace rocr {
namespace core {
class Queue;
}

namespace AMD {

class XdnaDriver final : public core::Driver {
public:
  XdnaDriver(std::string devnode_name);
  ~XdnaDriver();

  static hsa_status_t DiscoverDriver();

  /// @brief Returns the size of the dev heap in bytes.
  static uint64_t GetDevHeapByteSize();

  hsa_status_t Init() override;
  hsa_status_t QueryKernelModeDriver(core::DriverQuery query) override;

  hsa_status_t GetHandleMappings(std::unordered_map<uint32_t, void*> &vmem_handle_mappings);
  hsa_status_t GetAddrMappings(std::unordered_map<void*, uint32_t> &vmem_addr_mappings);
  hsa_status_t GetFd(int &fd);

  hsa_status_t GetAgentProperties(core::Agent &agent) const override;
  hsa_status_t
  GetMemoryProperties(uint32_t node_id,
                      core::MemoryRegion &mem_region) const override;
  hsa_status_t AllocateMemory(const core::MemoryRegion &mem_region,
                              core::MemoryRegion::AllocateFlags alloc_flags,
                              void **mem, size_t size,
                              uint32_t node_id) override;
  hsa_status_t FreeMemory(void *mem, size_t size) override;

  /// @brief Creates a context on the AIE device for this queue.
  /// @param queue Queue whose on-device context is being created.
  /// @return hsa_status_t
  hsa_status_t CreateQueue(core::Queue &queue) const override;
  hsa_status_t DestroyQueue(core::Queue &queue) const override;

  hsa_status_t ConfigHwCtx(core::Queue &queue,
                           hsa_amd_queue_hw_ctx_config_param_t config_type,
                           void *args) override;

private:
  hsa_status_t QueryDriverVersion();
  /// @brief Allocate device accesible heap space.
  ///
  /// Allocate and map a buffer object (BO) that the AIE device can access.
  hsa_status_t InitDeviceHeap();
  hsa_status_t FreeDeviceHeap();

  /// TODO: Remove this in the future and rely on the core Runtime
  /// object to track handle allocations. Using the VMEM API for mapping XDNA
  /// driver handles requires a bit more refactoring. So rely on the XDNA driver
  /// to manage some of this for now.
  std::unordered_map<uint32_t, void *> vmem_handle_mappings;
  std::unordered_map<void*, uint32_t> vmem_addr_mappings;

  /// @brief Configures the CUs associated with the HW context for this queue.
  ///
  /// @param config_cu_param CU configuration information.
  hsa_status_t
  ConfigHwCtxCU(core::Queue &queue,
                hsa_amd_aie_ert_hw_ctx_config_cu_param_addr_t &config_cu_param);

  /// @brief Virtual address range allocated for the device heap.
  ///
  /// Allocate a large enough space so we can carve out the device heap in
  /// this range and ensure it is aligned to 64MB. Currently, AIE2 supports
  /// 48MB device heap and it must be aligned to 64MB.
  void *dev_heap_parent = nullptr;

  /// @brief The aligned device heap.
  void *dev_heap_aligned = nullptr;
  static constexpr size_t dev_heap_size = 64 * 1024 * 1024;
  static constexpr size_t dev_heap_align = 64 * 1024 * 1024;
};

} // namespace AMD
} // namespace rocr

#endif // header guard
