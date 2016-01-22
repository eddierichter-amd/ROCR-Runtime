/*
 * Copyright 2015 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "hsakmt.h"
#include "amdp2ptest.h"

int rdma_fd = -1;

void rdma_open()
{
    rdma_fd = open(AMDP2PTEST_DEVICE_PATH, O_RDWR);

    if (-1 == rdma_fd ) {
        int ret = errno;
        fprintf(stderr, "error opening driver (errno=%d/%s)\n", ret, strerror(ret));
        exit(EXIT_FAILURE);
    }
}

void rdma_close()
{
    int retcode = close(rdma_fd);

    if (-1 == retcode) {
        fprintf(stderr, "error closing driver (errno=%d/%s)\n", retcode, strerror(retcode));
        exit(EXIT_FAILURE);
    }

    rdma_fd = -1;
}

int rdma_map(uint64_t gpu_ptr, size_t size, void **cpu_ptr)
{
    int ret = 0;

    *cpu_ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, rdma_fd, gpu_ptr);

    if (*cpu_ptr == NULL) {
        int __errno = errno;
        *cpu_ptr = NULL;
        fprintf(stderr, "Can't BAR, error=%s(%d) size=%zu offset=%llx\n",
                strerror(__errno), __errno, size, (long long unsigned)gpu_ptr);
        ret = __errno;
    }

    return ret;
}

int rdma_unmap(void *cpu_ptr, size_t size)
{
    int ret = 0;

    int retcode = munmap(cpu_ptr, size);

    if (-1 == retcode) {
        int __errno = errno;
        fprintf(stderr, "can't unmap BAR, error=%s(%d) size=%zu\n",
                strerror(__errno), __errno, size);
        ret = __errno;
    }

    return ret;
}

void run_rdma_tests(HSAuint32 Node, HsaMemoryProperties *MemoryProperty)
{
    printf("Size 0x%lx (%ld MB)\n", MemoryProperty->SizeInBytes,
                                        MemoryProperty->SizeInBytes / (1024 * 1024));
    printf("VirtualBaseAddress 0x%lx\n", MemoryProperty->VirtualBaseAddress);


    void *cpu_ptr;
    int ret = 0;
    void *MemoryAddress = 0;
    HSAuint64 SizeInBytes = 4096;
    HsaMemFlags memFlags = {0};

    memFlags.ui32.NonPaged    = 1;
    memFlags.ui32.CachePolicy = HSA_CACHING_WRITECOMBINED;
    memFlags.ui32.NoSubstitute = 1;
    memFlags.ui32.PageSize     = HSA_PAGE_SIZE_4KB;
//    memFlags.ui32.HostAccess   = 1;
    memFlags.ui32.CoarseGrain  = 1;

    HSAKMT_STATUS status = hsaKmtAllocMemory(Node,
                                             SizeInBytes,
                                             memFlags,
                                             &MemoryAddress);

    if (status != HSAKMT_STATUS_SUCCESS)
    {
        fprintf(stderr, "Failure to allocate memory. Status %d\n", status);
        exit(EXIT_FAILURE);
    }

    printf("Memory allocated. Address 0x%p\n", MemoryAddress);

    struct AMDRDMA_IOCTL_GET_PAGE_SIZE_PARAM get_page_size = {0};
    get_page_size.addr   = (uint64_t) MemoryAddress;
    get_page_size.length = SizeInBytes;

    ret = ioctl(rdma_fd, AMD2P2PTEST_IOCTL_GET_PAGE_SIZE, &get_page_size);

    if (ret != 0)
    {
        fprintf(stderr,
                "AMD2P2PTEST_IOCTL_GET_PAGE_SIZE error (errno=%d/%s)\n",
                ret, strerror(ret));
        exit(EXIT_FAILURE);
    }

    printf("GPU Page size: 0x%ld\n", get_page_size.page_size);

    struct AMDRDMA_IOCTL_GET_PAGES_PARAM get_cpu_ptr = {0};
    get_cpu_ptr.addr    = (uint64_t) MemoryAddress;
    get_cpu_ptr.length  = SizeInBytes;

    ret = ioctl(rdma_fd, AMD2P2PTEST_IOCTL_GET_PAGES, &get_cpu_ptr);

    if (ret != 0)
    {
        fprintf(stderr, "AMD2P2PTEST_IOCTL_GET_PAGES error (errno=%d/%s)\n",
                         ret, strerror(ret));
        exit(EXIT_FAILURE);
    }


    ret = rdma_map((uint64_t)MemoryAddress, 4096, &cpu_ptr);

    if (ret < 0)
    {
        exit(EXIT_FAILURE);
    }

    printf("CPU Virtual address 0x%p\n", cpu_ptr);

    hsaKmtFreeMemory(MemoryAddress, SizeInBytes);
}

int main(void)
{
    HsaVersionInfo      VersionInfo;

    HSAKMT_STATUS          status = hsaKmtOpenKFD();

    if( status == HSAKMT_STATUS_SUCCESS)
    {
        status = hsaKmtGetVersion(&VersionInfo);

        if(status == HSAKMT_STATUS_SUCCESS)
        {
            printf("Kernel Interface Major Version: %d\n", VersionInfo.KernelInterfaceMajorVersion);
            printf("Kernel Interface Minor Version: %d\n", VersionInfo.KernelInterfaceMinorVersion);
        }
    }

    rdma_open();

    HsaSystemProperties SystemProperties = {0};
    status = hsaKmtAcquireSystemProperties(&SystemProperties);

    if(status != HSAKMT_STATUS_SUCCESS)
    {
        fprintf(stderr, "hsaKmtAcquireSystemProperties call failed. Error: %d\n", status);
        exit(EXIT_FAILURE);
    }

    printf("System properties: Number of nodes: %d\n", SystemProperties.NumNodes);

    for (HSAuint32 iNode = 0; iNode < SystemProperties.NumNodes; iNode++)
    {
        HsaNodeProperties  NodeProperties = {0};
        status = hsaKmtGetNodeProperties(iNode, &NodeProperties);

        if(status != HSAKMT_STATUS_SUCCESS)
        {
            fprintf(stderr, "hsaKmtGetNodeProperties (Node = %d) call failed. Error: %d\n",
                             iNode, status);
            exit(EXIT_FAILURE);
        }

        printf("Node %d -> Number of Memory Banks = %d\n", iNode,
                            NodeProperties.NumMemoryBanks);

        HsaMemoryProperties*  MemoryProperties =
                    new HsaMemoryProperties[NodeProperties.NumMemoryBanks];

        status = hsaKmtGetNodeMemoryProperties(iNode,
                                               NodeProperties.NumMemoryBanks,
                                               MemoryProperties);

        if(status != HSAKMT_STATUS_SUCCESS)
        {
            fprintf(stderr, "hsaKmtGetNodeMemoryProperties (Node = %d) call failed. Error: %d\n",
                             iNode, status);
            exit(EXIT_FAILURE);
        }

        for (HSAuint32 iMemBank = 0; iMemBank < NodeProperties.NumMemoryBanks; iMemBank++)
        {
            printf("Heap type: %d\n", MemoryProperties[iMemBank].HeapType);

            if (MemoryProperties[iMemBank].HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC)
            {
                // We found local memory available for RDMA operation.
                // Run some tests on it.
                run_rdma_tests(iNode, &MemoryProperties[iMemBank]);
            }
        }
    }


    status = hsaKmtReleaseSystemProperties();

    if(status != HSAKMT_STATUS_SUCCESS)
    {
        fprintf(stderr, "hsaKmtReleaseSystemProperties call failed. Error: %d\n",
                status);
        exit(EXIT_FAILURE);
    }

    rdma_close();

    status = hsaKmtCloseKFD();

    if(status != HSAKMT_STATUS_SUCCESS)
    {
        fprintf(stderr, "hsaKmtCloseKFD call failed. Error: %d\n", status);
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}


