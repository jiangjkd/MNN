//
//  MetalBackend.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalBackend_hpp
#define MetalBackend_hpp

#include "core/Backend.hpp"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "MetalDefine.h"
#include <MNN/ErrorCode.hpp>
#include <vector>
#include <list>
#include <algorithm>
//#include "MNNMetalContext.h"
#include "MetalCache_generated.h"
using namespace MetalCache;

#if MNN_METAL_ENABLED
namespace MNN {

/** MetalRuntime */
enum MetalTuneLevel {Never = 0, Heavy = 1, Wide = 2, Normal = 3, Fast = 4};

struct MemChunkInfo {
    Tensor* t;
    size_t begin;
    size_t end;
    bool bOverWrite = false;
public:
    MemChunkInfo(Tensor* _t, size_t _begin, size_t _end ) {
        t = _t;
        begin = _begin;
        end = _end;
        bOverWrite = false;
    }
    size_t size() const {
        return end - begin;
    }
};

struct TunedInfo;
class MetalRuntime : public Runtime {
public:
    friend class MetalBackend;
    virtual ~ MetalRuntime();
    
    void *context() const {
        return mContext;
    }

    void setGpuMode(const int cl_mode_num);
    
    std::pair<const void*, size_t> makeCache(TunedInfo* info);
    bool setCache(std::pair<const void*, size_t> cache);
    
    MetalTuneLevel getTuneLevel() {
        return mTuneLevel;
    }
    std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>,  uint32_t>>& getTunedThreadGroup() {
        return mTunedThreadGroup;
    };
    virtual Backend *onCreate(const BackendConfig* config) const override;
    virtual void onGabageCollect(int level) override;
    virtual CompilerType onGetCompilerType() const override {
        return Compiler_Loop;
    }
    virtual float onGetMemoryInMB() override;

    virtual std::pair<const void*, size_t> onGetCache() override;
    virtual bool onSetCache(const void* buffer, size_t size) override;

    static MetalRuntime* create(const Backend::Info& info, id<MTLDevice> device);
    virtual void onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const MNN::Op* op) override;
    virtual bool onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const MNN::Op* op, Runtime::OpInfo& dstInfo) const override;
private:
    MetalRuntime(void* context);
    void* mContext = nullptr;
    std::shared_ptr<EagerBufferAllocator> mStatic;
    MetalTuneLevel mTuneLevel = Wide;
    std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, uint32_t>> mTunedThreadGroup;

private:
    std::vector<uint8_t> mBuffer;
    const void* mCacheOutside = nullptr;
    size_t mCacheOutsideSize = 0;
    TunedInfo* mTunedInfo;
};


class MetalRuntimeAllocator : public EagerBufferAllocator::Allocator {
public:
    class MetalBufferAlloc {
    public:
        MetalBufferAlloc(id<MTLBuffer> buffer) {
            mBuffer = buffer;
        }
        id<MTLBuffer> getBuffer() {
            return mBuffer;
        }
        ~MetalBufferAlloc(){
            mBuffer = nil;
        };
    private:
        id<MTLBuffer> mBuffer = nil;
    };
    
    MetalRuntimeAllocator(id<MTLDevice> device): mDevice(device) {
        // Do nothing
    }
    virtual ~ MetalRuntimeAllocator() = default;
    virtual MemChunk onAlloc(size_t size, size_t align) override;
    virtual void onRelease(MemChunk ptr) override;
    
private:
    id<MTLDevice> mDevice;
};

/** Metal backend */
class MetalBackend : public Backend {
public:
    /** Metal execution creator */
    class Creator {
    public:
        /**
         * @brief create execution for given input, op on metal backend.
         * @param inputs    given input tensors.
         * @param op        given op.
         * @param backend   metal backend.
         * @return created execution if supported, NULL otherwise.
         */
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *> &outputs) const = 0;
    };
    /**
     * @brief register creator for given op type.
     * @param type      given op type.
     * @param creator   registering creator.
     */
    static void addCreator(OpType type, Creator *creator);

    id<MTLBuffer> getHostBuffer(size_t size) const;
    id<MTLBuffer> getConstBuffer(size_t size) const;
public:
    MetalBackend(std::shared_ptr<EagerBufferAllocator> staticMem, const MetalRuntime* runtime);
    virtual ~MetalBackend();
    const MetalRuntime* runtime() const {
        return mRuntime;
    }
    
    virtual Backend::MemObj* onAcquire(const Tensor *Tensor, StorageType storageType, const class Tensor *owTensor = nullptr) override;
    virtual void onAllocFromStaticPlan(const Tensor *Tensor) override;
    virtual void onRemoveTempStaticPlan(const Tensor *Tensor) override;
    virtual bool onRelease(const Tensor* tensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual void onClearPoolStatic() override;
    virtual void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor) const override;

    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) override;

    virtual void onStaticMemPlanBegin() override;
    virtual void onStaticMemPlanEnd() override;

    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    virtual int onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) override;

public:
    /**
     * @brief get metal context object
     * @return metal context object pointer
     */
    void *context() const;

    /**
     * @brief copy buffer content to dest tensor
     * @param srcTensor source tensor
     * @param dstTensor destined tensor
     * @param encoder command encoder
     */
    void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor,
                              id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape) const;

    void flushEncoder() const;
    id<MTLComputeCommandEncoder> encoder() const;
    void addOpEncoder(std::function<void(void)> opEncoder);
    
    bool isCommandEncoderSet();
    void setOpEncoder() const;
    
    EagerBufferAllocator *getBufferPool() const {
        return mBufferPool.get();
    }
    EagerBufferAllocator *getStaticBufferPool() const {
        return mStaticBufferPool.get();
    }

    bool isCmdBufferCommit();

    bool isBigMode() {return mBigMode;}
private:

    std::list<MemChunkInfo> mUseChunkInfoList;
    std::list<MemChunkInfo> mFreeChunkInfoList;
    //start for mem static plan mode
    bool mBigMode = false;
    std::unordered_map<const Tensor*,  std::tuple<size_t, size_t>> mTensorChunkInfoMap;
    size_t mStaticPlanSize = 0;
    // for small block mode
    std::unordered_map<const Tensor*,  std::tuple<size_t, size_t>> mSmallModeTensorChunkInfoMap;
    size_t mSmallModeStaticPlanSize = 0;


    Backend::MemObj* mStaticPlanMem = nullptr;

    //end

    const MetalRuntime* mRuntime;
    std::vector<id<MTLBuffer>> mHoldBuffers;
    id<MTLBuffer> mShapeH2D;
    id<MTLBuffer> mShapeD2H;
    mutable NSUInteger mEncoderCount = 0;
    mutable bool mOpEncoderSet = false;//whether has set encoder
    mutable bool mOpFullSupport = true;
    mutable bool mFrameEncodeCache = false;

    std::vector<std::function<void(void)>> mOpEncoders;
    mutable id<MTLComputeCommandEncoder> mComputeEncoder = nil;
    std::shared_ptr<EagerBufferAllocator> mBufferPool;
    std::shared_ptr<EagerBufferAllocator> mStaticBufferPool;

private:
    void onAcquireStaticPlan(const Tensor *Tensor, size_t size, const class Tensor *owTenser = nullptr);
    void onReleaseStaticPlan(const Tensor* tensor);
    mutable id<MTLBuffer> mHostBuffer = nullptr;
    void onCopyHostToDevice(const Tensor *src, const Tensor *dst) const;
    void onCopyDeviceToHost(const Tensor *src, const Tensor *dst) const;
    void onCopyDeviceToDevice(const Tensor *src, const Tensor *dst, id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape) const;
};


/** Metal creator register */
template <class T>
class MetalCreatorRegister {
public:
    /**
     * @brief initializer. register T creator for given op type.
     * @param type  given op type.
     */
    MetalCreatorRegister(OpType type) {
        T *test = new T;
        MetalBackend::addCreator(type, test);
    }
};
} // namespace MNN

#define REGISTER_METAL_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {              \
        MetalBackend::addCreator(opType, new name); \
    }

#endif /* MNN_METAL_ENABLED */
#endif /* MetalBackend_hpp */
