//
//  MetalRaster.hpp
//  MNN
//
//  Created by MNN on 2020/05/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalRaster_hpp
#define MetalRaster_hpp

#import "core/Execution.hpp"
#import "MetalDefine.h"
#include <map>

#if MNN_METAL_ENABLED
namespace MNN {

class MetalRaster : public Execution {
public:
    MetalRaster(Backend *backend);
    virtual ~MetalRaster();
    virtual ErrorCode onResizeStaticMemPlan(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    void _scheduleStaticMemPlan(const Tensor* output, std::map<Tensor*, std::shared_ptr<Tensor>>& tempInput, std::shared_ptr<Tensor>& tempOutput);
private:
    std::map<Tensor*, std::shared_ptr<Tensor>> mTempInput;
    std::shared_ptr<Tensor> mTempOutput;

    std::map<Tensor*, std::shared_ptr<Tensor>> mSmallModeTempInput;
    std::shared_ptr<Tensor> mSmallModeTempOutput;

    std::vector<std::tuple<id<MTLBuffer>, id<MTLBuffer>, MTLSize, MTLSize, int> > mTempInputCopy;
    bool mNeedZero = false;
    id<MTLBuffer> mOutputPtr;
    bool mFast = false;
    id<MTLComputePipelineState> mBlitPipeline;
    std::vector<id<MTLBuffer>> mShapeTemp;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalRaster_hpp */
