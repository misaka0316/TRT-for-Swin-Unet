#include "LayerNorm.h"

__global__ void layerNormGPU(const float *pInput, float *pOutput，int dim , int dim2)
{
    // const int tx = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tx = threadIdx.x;
    __shared__ float temp[1024];
    __shared__ float var[2048];
    __shared__ double m [17];
    double square0, square1;
    //加载输出到shared memory
    var[tx] = pInput[idx];
    __syncthreads();
    //元素和
    if (tx < dim2)
    {
        temp[tx] = var[tx] + var[tx + dim];
    }
    __syncthreads();

    for (int stride = dim2; stride >= 3; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] = temp[tx]  + temp[tx + stride];
        }
        __syncthreads();
    }
    //求平均值
    if (tx < 1)
    {
        temp[0] = temp[0] + temp[1] + temp[2];
        m[0] = temp[0] / dim;
    }
    __syncthreads();
    
    //平方和
    if (tx < dim2)
    {   
        square0 = var[tx] - m[0];
        square1 = var[tx + dim2] - m[0];
        temp[tx] = square0 * square0 + square1 * square1;   //平方期望
    }
    __syncthreads();

    for (int stride = dim2; stride >= 3; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] = temp[tx] + temp[tx + stride];
        }
        __syncthreads();
    }
    //求平方和平均值
    if (tx < 1)
    {
    temp[0] = temp[0] + temp[1] + temp[2];
    m[16] = temp[0] / dim;
    }
    __syncthreads();

    pOutput[idx] = (var[tx] - m[0]) * rsqrtf(m[16] + 0.000009999999747378752);
    // pOutput[threadIdx.x + 128] = (pInput[tx + 128] - mean) * rsqrtf(var + 1e-6);
}


namespace nvinfer1 {
    LayerNormPlugin::LayerNormPlugin (const std::string &name): name_(name) {
        WHERE_AM_I()
    }

    LayerNormPlugin::~LayerNormPlugin () {
        WHERE_AM_I()
    }

    IPluginV2DynamicExt *LayerNormPlugin::clone() const noexcept {
        WHERE_AM_I()
        auto p = new LayerNormPlugin(name_);
        p->setPluginNamespace(namespace_.c_str());
        return p;
    }

    int32_t LayerNormPlugin::getNbOutputs() const noexcept {
        WHERE_AM_I()
        return 1;
    }

    DataType LayerNormPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept {
        WHERE_AM_I()
        return inputTypes[0];
    }

    DimsExprs LayerNormPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept {
        WHERE_AM_I()
        return inputs[0];
    }

    bool LayerNormPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
        WHERE_AM_I()
        switch (pos)
        {
            case 0:
                return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
            case 1:
                return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
            default: // should NOT be here!
                return false;
        }
        return false;
    }

    void LayerNormPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
        WHERE_AM_I()
    }

    size_t LayerNormPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
        WHERE_AM_I()
        printf("batch size = %d\n",inputs[0].dims.d[0]);
        return 0;
    }

    int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
        WHERE_AM_I()
        int dim2 = inputDesc[0].dims.d[2] / 2;
        layerNormGPU<<< inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1], inputDesc[0].dims.d[2]>>>(reinterpret_cast<const float * >(inputs[0]), reinterpret_cast<float * >(outputs[0]), inputDesc[0].dims.d[2], dim2);
        return 0;
    }

    void LayerNormPlugin::destroy() noexcept {
        WHERE_AM_I()
        //delete this;
    }

    int32_t LayerNormPlugin::initialize() noexcept {
        WHERE_AM_I()
        return 0;
    }

    void LayerNormPlugin::terminate() noexcept {
        WHERE_AM_I()}

    size_t LayerNormPlugin::getSerializationSize() const noexcept {
        WHERE_AM_I()
        return 0;
    }

    void LayerNormPlugin::serialize(void *buffer) const noexcept {
        WHERE_AM_I()
    }

    void LayerNormPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
        WHERE_AM_I()
        namespace_ = pluginNamespace;
    }

    const char *LayerNormPlugin::getPluginNamespace() const noexcept {
        WHERE_AM_I()
        return namespace_.c_str();
    }

    const char *LayerNormPlugin::getPluginType() const noexcept {
        WHERE_AM_I()
        return PLUGIN_NAME;
    }

    const char *LayerNormPlugin::getPluginVersion() const noexcept {
        WHERE_AM_I()
        return PLUGIN_VERSION;
    }

    void LayerNormPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept {
        WHERE_AM_I()
    }

    void LayerNormPlugin::detachFromContext() noexcept {
        WHERE_AM_I()
    }

// class AddScalarPluginCreator
    PluginFieldCollection LayerNormPluginCreator::fc_ {};
    std::vector<PluginField> LayerNormPluginCreator::attr_;

    LayerNormPluginCreator::LayerNormPluginCreator() {
        WHERE_AM_I()
        fc_.nbFields = attr_.size();
        fc_.fields   = attr_.data();
    }

    LayerNormPluginCreator::~LayerNormPluginCreator() {
        WHERE_AM_I()
    }

// 最重要的两个成员函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
    IPluginV2 *LayerNormPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept {
        WHERE_AM_I()
        return new LayerNormPlugin(name);
    }

    IPluginV2 *LayerNormPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept {
        WHERE_AM_I()
        return new LayerNormPlugin(name);
    }

    void LayerNormPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
        WHERE_AM_I()
        namespace_ = pluginNamespace;
    }

    const char *LayerNormPluginCreator::getPluginNamespace() const noexcept {
        WHERE_AM_I()
        return namespace_.c_str();
    }

    const char *LayerNormPluginCreator::getPluginName() const noexcept {
        WHERE_AM_I()
        return PLUGIN_NAME;
    }

    const char *LayerNormPluginCreator::getPluginVersion() const noexcept {
        WHERE_AM_I()
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection *LayerNormPluginCreator::getFieldNames() noexcept {
        WHERE_AM_I()
        return &fc_;
    }

    REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

} // namespace nvinfer1

