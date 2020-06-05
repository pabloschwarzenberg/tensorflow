#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c/c_api.h>

int Okay(TF_Status* status)
{
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: %s\n", TF_Message(status));
        return 0;
    }
    return 1;
}

int main()
{
    TF_Graph* Graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();
    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;

    const char* saved_model_dir = ".\\python\\predictor";
    const char* tags = "serve"; 
    int ntags = 1;
    printf("Inicializando tensorflow %s\n", TF_Version());

    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);

    //listar con saved_model_cli show --dir predictor --tag_set serve --signature_def serving_default

    int NumInputs = 1;
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);
    TF_Output x = { TF_GraphOperationByName(Graph, "dense_input"), 0 };
    Input[0] = x;

    int NumOutputs = 1;
    TF_Output* OutPut = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);
    TF_Output y = { TF_GraphOperationByName(Graph, "dense_1/BiasAdd"), 0 };
    OutPut[0] = y;

    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumInputs);
    TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumOutputs);

    int i;
    int ndims = 2;
    int size;
    int64_t input_dims[] = { 1,128 };
    float input_data[] = {
    0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };

    TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT,input_dims,ndims,sizeof(float)*128);
    size = TF_TensorByteSize(input_tensor);
    float* tensor_buffer = (float*)TF_TensorData(input_tensor);
    memcpy(tensor_buffer, input_data, size);

    InputValues[0] = input_tensor;
    OutputValues[0] = { NULL };

    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, OutPut, OutputValues, NumOutputs, NULL, 0, NULL, Status);

    if (Okay(Status))
    {
        int nbytes = sizeof(float) * 2;
        if (TF_TensorByteSize(OutputValues[0]) != nbytes) {
            fprintf(stderr,
                "ERROR: Expected predictions tensor to have %zu bytes, has %zu\n",
                nbytes, TF_TensorByteSize(OutputValues[0]));
            TF_DeleteTensor(OutputValues[0]);
            return 0;
        }
        float* predictions = (float*)malloc(nbytes);
        memcpy(predictions, TF_TensorData(OutputValues[0]), nbytes);
        TF_DeleteTensor(OutputValues[0]);

        printf("Predictions:\n");
        for (int i = 0; i < 2; ++i) {
            printf("\tpredicted y = %f\n", predictions[i]);
        }
        free(predictions);
    }

    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);

    return 0;
}