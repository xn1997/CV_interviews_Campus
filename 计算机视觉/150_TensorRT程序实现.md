# 搭建流程

## 模型转换及预测

### 通用函数编写

1. 定义一个G数据占用内存量

   ```python
   def GiB(val):
       return val * 1 << 30
   ```
   
2. 构建engine函数

   (pytorch逐层解析参考链接：

   https://gitee.com/xn1997/TensorRT-sample/blob/master/python/network_api_pytorch_mnist/model.py)

   ```python
   def build_engine(max_batch_size, onnx_file_path="", engine_file_path="", \
                    fp16_mode=False, int8_mode=False, save_engine=False):
       """Takes an ONNX file and creates a TensorRT engine to run inference with"""
       with trt.Builder(TRT_LOGGER) as builder, \
               builder.create_network(common.EXPLICIT_BATCH) as network, \
               trt.OnnxParser(network, TRT_LOGGER) as parser:  # 对于caffe和onnx可以直接使用parser，对于pytorch需要自己逐层解析
   
           "1. 设置builder一些属性"
           builder.max_workspace_size = common.GiB(1)  # 1 << 30  # Your workspace size
           builder.max_batch_size = max_batch_size
           builder.fp16_mode = fp16_mode  # Default: False
           builder.int8_mode = int8_mode  # Default: False
           if int8_mode:
               # To be updated
               raise NotImplementedError
   
           "2. 解析模型"
           print('Loading ONNX file from path {}...'.format(onnx_file_path))
           with open(onnx_file_path, 'rb') as model:
               print('Beginning ONNX file parsing')
               parser.parse(model.read())
           print('Completed parsing of ONNX file')
           "3. 构建engine"
           print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
           engine = builder.build_cuda_engine(network)
           print("Completed creating Engine")
   		"4. 保存engine"
           if save_engine:
               with open(engine_file_path, "wb") as f:
                   f.write(engine.serialize())
           return engine
   ```

3. 读入保存好的engine

   ```python
   def read_engine(trt_engine_path):
       with open(trt_engine_path, "rb") as f, \
               trt.Runtime(TRT_LOGGER) as runtime:
           engine = runtime.deserialize_cuda_engine(f.read())
       return engine
   ```

4. 为模型分配空间

   ```python
   class HostDeviceMem(object):
       def __init__(self, host_mem, device_mem):
           """Within this context, host_mom means the cpu memory and device means the GPU memory
           """
           self.host = host_mem
           self.device = device_mem
   
       def __str__(self):
           return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
   
       def __repr__(self):
           return self.__str__()
       
   def allocate_buffers(engine):
       inputs = []
       outputs = []
       bindings = []
       stream = cuda.Stream()
       "只有input和output两个元素，即只循环两次"
       for binding in engine:
           size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # 计算input大小(n*c*h*w)
           dtype = trt.nptype(engine.get_binding_dtype(binding))  # 获的该层的数据类型(np.float32)
           # Allocate host and device buffers  size*sizeof(dtype)=需要分配的内存数
           host_mem = cuda.pagelocked_empty(size, dtype)
           device_mem = cuda.mem_alloc(host_mem.nbytes)
           # Append the device buffer to device bindings.
           bindings.append(int(device_mem))
           # Append to the appropriate list.
           if engine.binding_is_input(binding):
               inputs.append(HostDeviceMem(host_mem, device_mem))
           else:
               outputs.append(HostDeviceMem(host_mem, device_mem))
       return inputs, outputs, bindings, stream
   ```

5. 推理函数

   ```python
   def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
       # Transfer data from CPU to the GPU.
       [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
       # Run inference.
       context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
       # Transfer predictions back from the GPU.
       [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
       # Synchronize the stream
       stream.synchronize()
       # Return only the host outputs.
       return [out.host for out in outputs]  # list：每个元素代表一个输出，每个输出包含多个batch
   ```

6. 后处理。将一维输出，reshape回多个batch

   ```python
   def postprocess_the_outputs(h_outputs, shape_of_output):
       h_outputs = h_outputs.reshape(*shape_of_output)  # 回忆一下*的用法
       return h_outputs
   ```

   

### 流程概述

定义`TRT_LOGGER`，用于转换过程中警告和错误的打印

```python
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  # This logger is required to build an engine
```

1. 生成engine

   ```python
   fp16_mode = False
   int8_mode = False
   trt_engine_path = './model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)
   # Build an engine
   engine = build_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode, save_engine=True)
   ```

2. 读入engine

   ```python
   engine = read_engine(trt_engine_path)
   ```

3. 为模型分配空间

   ```python
   inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host
   ```

4. 创建context

   ```python
   context = engine.create_execution_context()
   ```

5. 加载输入

   ```PYTHON
   inputs[0].host = img_np_nchw.reshape(-1) # 必须变成一维向量，无论有几个batch。如果有多个输入就逐个赋值给input[1]
   ```

   必须变成一维向量，且必须附给`inputs[0].host`，原因不明。

6. 推理

   ```python
   trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # 得到的是一个list，参见do_inference输出
   ```

7. 后处理

   ```python
   shape_of_output = (max_batch_size, 1000)  # max_batch_size和构建engine时一样
   # 输入的是一维向量，输出的也是一维，需要reshape到期望的batch大小
   feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)
   # 如果有多个输出(像yolo就有3个)则还应该继续解析输出
   feat1 = postprocess_the_outputs(trt_outputs[1], shape_of_output)
   ```


# 转换方案

## tkDNN（逐层解析）

1. 要导出每层网络结构及权重（涉及自定义层）
2. 解析程序也需要自己编写（按照网络结构，用tensorrt重新编写一遍，太麻烦了，工作量大不实际）
3. 推理需要自己编写（tkDNN很鸡肋，只能使用他做好的那几个网络，意义不大）
4. int8好做

## TensorRT（整体解析）

1. 转换到onnx（可能涉及不支持的层）
2. 转换到tensorrt（涉及简单的输入输出）
3. 推理方便
4. int8自己写（应该不难）

# pytorch-->ONNX-->TensorRT

## pytorch-->ONNX

```python
"1. 转换为onnx"
input_name = ['input']
output_name = ['output1', 'output0']  # 名字任意定，只要小于等于真实的输出数量即可
# output_name = ['output']  # 必须要有输入输出
input = (torch.randn(1, 3, 224, 224)).cuda()
model = Backbone(pretrained=True,num_classes=1000).cuda()
torch.onnx.export(model, input, 'resnet50.onnx', input_names=input_name, output_names=output_name, verbose=True)

"2. 验证onnx是否正确"
test = onnx.load('resnet50.onnx')
onnx.checker.check_model(test)
print("==> Passed")
```

### 常见错误

1. 用DataParallel保存的模型key值前面会多处’modules.’，而我们需要的不带，就导致生成onnx的方法无法读到对应的属性

   解决方案：参考链接：[解决pytorch中DataParallel后模型参数出现问题的方法](https://blog.csdn.net/genous110/article/details/88385940)

   ```python
   from collections import OrderedDict
   # 1. 从文件加载参数
   state_dict = torch.load(‘myfile.pth.tar’)
   
   new_state_dict = OrderedDict()
   for k, v in state_dict.items():
       name = k[7:] # remove module.
       new_state_dict[name] = v
   
   model.load_state_dict(new_state_dict)
   
   # 2. 模型中已经加载好了参数
   state_dict = model.state_dict()
   ... # 内容同上
   model = model.moudle
   model.load_state_dict(new_state_dict)
   ```

   

## ONNX-->TensorRT

参见1.1.2的第1步，engine构建即可。

### output_name的输出顺序如何确定

- 根据output_name的顺序和实际网络输出的顺序决定

如：

`output_name = ['output1', 'output0'] `

```python

        conv4 = x[:, 0, :, :]  # (1,7,7)

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # (1,1000)

        return x, conv4
```

1. onnx中

   ```python
   output1 = x  # (1,1000)
   output0 = conv4  # (1,7,7)
   # 即二者顺序一致
   ```

2. TensorRT中

   ```python
   # 和onnx中的输出名字没有关系，直接将输出放在一个list中
   # list的顺序即是输出的顺序，假如输出保存在output变量中，则
   output[0] = conv4  # (1,7,7)
   output[1] = x  # (1,1000)
   ```

   **tensorrt的输出是根据网络输出顺序定义的，先输出的就靠前**

   - 强制改变顺序的小trick

     ```python
     # 修改网络结构
     return x, conv4*1
     ```

     这样就会多增加一个层，顺序由最后返回的顺序决定，但会增加计算量，没必要这么做。

# ONNX-->TensorRT(C++)

参考链接：[如何使用TensorRT对训练好的PyTorch模型进行加速?](https://zhuanlan.zhihu.com/p/88318324)(重点参考python版本，C++程序就是tensorrt官方代码可以看看)

[Mnist onnx model to c++ Tensor RT](https://blog.csdn.net/u012235274/article/details/103786817)(C++在其版本上修改，使其可以多输入多输出)

## 通用函数编写

1. 内存释放函数（`cuda`和`trt中的各种类`）

   ```c++
   struct InferDeleter
   {
     template <typename T>
     void operator()(T* obj) const{
       if (obj)
       {
           obj->destroy();
       }
     }
   };
   
   struct CudaDeleter
   {
     void operator()(void* obj){
       if (obj)
       {
           cudaFree(obj);
       }
     }
   };
   ```

2. 读取engine---`loadEngine`

   ```cpp
   ICudaEngine* loadEngine(const std::string& engine)
   { //输入engine文件路径即可
       std::ifstream engineFile(engine, std::ios::binary);
       if (!engineFile)
       {
           std::cout << "Error opening engine file: " << engine << std::endl;
           return nullptr;
       }
   
       engineFile.seekg(0, engineFile.end); //! 将读指针定位至文件末尾
       long int fsize = engineFile.tellg(); //! 读取当前指针的位置，即文件的大小
       engineFile.seekg(0, engineFile.beg); //! 将读指针重新移至文件开头
   
       std::vector<char> engineData(fsize);
       engineFile.read(engineData.data(), fsize);
       if (!engineFile)
       {
           std::cout << "Error loading engine file: " << engine << std::endl;
           return nullptr;
       }
   
       std::unique_ptr<IRuntime,InferDeleter> runtime(createInferRuntime(gLogger));
   
       return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
   }
   ```

3. 分配空间---`allocate_buffers()`

   ```cpp
   void allocate_buffers(std::unique_ptr<ICudaEngine, InferDeleter> &engine,int max_batch_size,
                  std::vector<int> &inputIndex,CPU_data &input_cpu_data, CPU_data &output_cpu_data,void** buffers)
   {
       // 3.2 分配输入、输出内存（cpu+gpu）
       int NbBindings = engine->getNbBindings();  // number of input+output
   //    void* buffers[NbBindings];  // initialize buffers(for gpu data)
   
       for (int i = 0; i < NbBindings; i++)
       {
           auto dims = engine->getBindingDimensions(i);
           size_t vol = static_cast<size_t>(max_batch_size);
           DataType type = engine->getBindingDataType(i);
           vol *= samplesCommon::volume(dims);
           size_t size_binding = vol * samplesCommon::getElementSize(type);
   
           cudaMalloc(&buffers[i], size_binding);  // allocate gpu memery
           std::vector<float> temp_data(vol);
           bool is_input = engine->bindingIsInput(i);
           if(is_input){ // 分配
               inputIndex.push_back(i);
               input_cpu_data.push_back(temp_data);  // 创建cpu输入
               input_cpu_data.size.push_back(size_binding);  // 记录输入占用字节数
           }
           else {
               output_cpu_data.push_back(temp_data);
               output_cpu_data.size.push_back(size_binding);
           }
       }
       return;
   }
   ```

4. cv::Mat图片转换为1维向量，以便输入网络---`normal()`

   ```cpp
   static const float kMean[3] = { 0.485f, 0.456f, 0.406f };
   static const float kStdDev[3] = { 0.229f, 0.224f, 0.225f };
   static const int map_[7][3] = { {0,0,0} ,
   				{128,0,0},
   				{0,128,0},
   				{0,0,128},
   				{128,128,0},
   				{128,0,128},
   				{0,128,0}};
   
   
   float* normal(cv::Mat img) {
       //将cv::Mat格式的图片,转换成一维float向量
       float * data = (float*)calloc(img.rows*img.cols * img.channels(), sizeof(float));
   //    printf("image channel %d\n",img.channels());
       if(img.channels()==3){
           for (int c = 0; c < 3; ++c)
           {
               for (int i = 0; i < img.rows; ++i)
               { //获取第i行首像素指针
                   cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
                   for (int j = 0; j < img.cols; ++j)
                   {
                       data[c * img.cols * img.rows + i * img.cols + j] = (p1[j][c] / 255.0f - kMean[c]) / kStdDev[c];
                   }
               }
           }
       }
       else if(img.channels()==1){
           for (int c = 0; c < 1; ++c)
           {
               for (int i = 0; i < img.rows; ++i)
               { //获取第i行首像素指针
                   cv::Vec<uchar,1> *p1 = img.ptr<cv::Vec<uchar,1>>(i);
                   for (int j = 0; j < img.cols; ++j)
                   {
                       data[c * img.cols * img.rows + i * img.cols + j] = p1[j][c];
                   }
               }
           }
       }
       else{
           printf("!!!!!!!!!!!!!!!!!!图片输入错误\n");
       }
   	return data;
   }
   ```

5. 推理

   ```cpp
   void trt_infer(cudaStream_t &stream,std::unique_ptr<IExecutionContext, InferDeleter> &context,int max_batch_size,std::vector<int> &inputIndex,CPU_data &input_cpu_data, CPU_data &output_cpu_data,void** buffers)
   {
       auto start_time = std::chrono::system_clock::now();
       for(int i=0; i<inputIndex.size();++i){
           cudaMemcpyAsync(buffers[inputIndex[i]],input_cpu_data[i].data(),input_cpu_data.size[i],cudaMemcpyHostToDevice,stream);
       }
       //  context->enqueue(max_batch_size,buffers,stream,nullptr);
       context->execute(max_batch_size,buffers);
       for(int i=0; i<output_cpu_data.data()->size();i++){
           cudaMemcpyAsync(output_cpu_data[i].data(), buffers[i+inputIndex.size()], output_cpu_data.size[i], cudaMemcpyDeviceToHost, stream);
       }
       cudaStreamSynchronize(stream);
       auto end_time = std::chrono::system_clock::now();
       std::cout << "infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;
   }
   ```

6. 后处理函数（1d输出reshape到2d）

   ```c++
   std::vector<std::vector<float>> reshape_1to2D(std::vector<int> shape,std::vector<float> data){
       std::vector<std::vector<float>> output(shape[0]);
       for(int i=0; i<output.size();++i)
           output[i].resize(shape[1]);
   
       for(int i=0; i<shape[0];++i){
           for(int j=0; j<shape[1]; ++j){
               output[i][j] = data[i*shape[1]+j];
           }
       }
       return output;
   }
   // 打印输出（10×10）
   void printfVector2D(std::vector<std::vector<float>> arrays,int max_lengths){
       for(int i = 0; i < arrays.size() && i<max_lengths; ++i) {
           for(int j = 0; j < arrays[i].size() && j < max_lengths; ++j) {
               std::cout << arrays[i][j] << " ";
           }
           std::cout << "\n";
       }
   }
   ```

   

## 流程

### 可调整参数

```cpp
int max_batch_size = 1;
int INPUT_H=224,INPUT_W=224,INPUT_C=3;
std::vector<std::vector<int>> OUTPUT_SIZE{{7,7},{1,1000}};  // for reshape output
```

### 定义`Logger`和`explicitBatch`

```cpp
using namespace nvinfer1;
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
        std::cout << msg << std::endl;
    }
};
extern Logger gLogger;

const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
```

### 生成engine

```cpp
    std::unique_ptr<IBuilder, InferDeleter> builder(createInferBuilder(gLogger));
    builder->setMaxBatchSize(1);

    std::unique_ptr<INetworkDefinition, InferDeleter> network(builder->createNetworkV2(explicitBatch));
    std::unique_ptr<nvonnxparser::IParser, InferDeleter> parser(nvonnxparser::createParser(*network, gLogger));
    parser->parseFromFile(onnxModelFile, static_cast<int>(ILogger::Severity::kWARNING));
	// 所有的配置尽量在config中设置（如FP16等）
    std::unique_ptr<IBuilderConfig, InferDeleter> config(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1_GiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
//    config->setFlag(BuilderFlag::kFP16);

    std::unique_ptr<ICudaEngine, InferDeleter> engine(builder->buildEngineWithConfig(*network, *config));
```

### 保存/读取`engine`

```cpp
// 保存engine
	IHostMemory* modelStream = engine->serialize();
    std::ofstream p(engine_save_path,std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    printf("modelStream->size():%d\n", modelStream->size());
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
// 读取engine
	std::unique_ptr<ICudaEngine, InferDeleter> engine(loadEngine(engine_path));
```

### 创建`context`

```cpp
std::unique_ptr<IExecutionContext, InferDeleter> context(engine->createExecutionContext());
```

### 加载输入

```cpp
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    cv::Mat dst = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
    cv::resize(img,dst, dst.size());
    float* fileData=normal(dst);
    for(int i = 0; i < INPUT_H*INPUT_W*INPUT_C; ++i){
        input_cpu_data[0][i] = fileData[i];
    }
    free(fileData);  // 释放图片
```

### 推理

```c++
trt_infer(stream,context,max_batch_size,inputIndex,input_cpu_data,output_cpu_data,buffers);
//    bool is_success = context->execute(1,buffers);
//    bool is_success = context->enqueue(1,buffers,stream,nullptr);
//    bool is_success = context->executeV2(buffers);
//    bool is_success = context->enqueueV2(buffers,stream,nullptr);
```

这四句一样，建议使用`enqueue`，这样和`python`版的`execute_async`输入一致。

### 输出调整

- 将一维输出reshape到常用的2d

```cpp
    for(int i = 0; i < OUTPUT_SIZE.size(); ++i){
        std::vector<std::vector<float>> a = reshape_1to2D(OUTPUT_SIZE[i],output_cpu_data[i]);
        printfVector2D(a,10);
    }
//    cudaStreamDestroy(stream); // 如果程序结束一定要释放
```

# 加速效果实验

## 结论

PyTorch ---> TensorRT：3倍多

FP32 ---> int8：接近3倍

30ms左右才能算正常

## HigerHRNet

model size：115.2M

|      | size      | precision | PyTorch | TensorRT | ratio |
| ---- | --------- | --------- | ------- | -------- | ----- |
|      | (256,256) | FP32      | 48.94   | 15.45    | 3.168 |
|      | (512,512) | FP32      | 78.58   | 49.98    | 1.572 |

检测结果示例：<img src="https://raw.githubusercontent.com/xn1997/picgo/master/HigherHRNet512_result.jpg" style="zoom:25%;" />

OpenPose size: 104.7M（～150ms）

|                                 | size(H,W) | precision | TensorRT |
| ------------------------------- | --------- | --------- | -------- |
| OpenPose                        | (480,640) | FP32      | 133      |
| HigerHRNet                      | (512,512) | FP32      | 49.98    |
| ratio                           | 1.172     |           | 2.66     |
| HigerHRNet(+upsample后处理部分) | (512,512) | FP32      | 60.23    |

## YOLOv4（tkDNN）

model size：256M

| size      | precision | OpenCV(CPU) | tkDNN | ratio |
| --------- | --------- | ----------- | ----- | ----- |
| (412,412) | FP32      | 223.2       | 42.8  | 5.215 |
|           | INT8      |             | 22.5  | 1.902 |
| (224,224) | FP32      | 90.2        | 14.8  | 6.095 |
|           | INT8      |             | 10.3  | 1.437 |

## OpenPose(Jetson)

20~21ms

FP32和FP16没有变化（原因不明，但jetson确实支持FP16）

| Size      | precision | TensorRT | ratio |
| --------- | --------- | -------- | ----- |
| (480,640) | FP32      | 200      |       |
| (224,224) | FP32      | 46.95    | 4.26  |
| (224,224) | INT8      | 20.4     | 2.3   |
| (112,112) | FP32      | 20       |       |

