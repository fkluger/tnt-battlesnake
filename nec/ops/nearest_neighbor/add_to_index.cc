#include <string>
#include <iostream>

#include <faiss/IndexFlat.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"

using namespace tensorflow;

class IndexContainer : public ResourceBase
{
public:
  faiss::IndexFlatL2 index;
  explicit IndexContainer(int d)
  {
    index = faiss::IndexFlatL2(d);
  }
  virtual string DebugString()
  {
    return std::to_string(this->index.ntotal);
  }
};

class AddToIndexOp : public OpKernel
{
public:
  explicit AddToIndexOp(OpKernelConstruction *context) : OpKernel(context)
  {
    OP_REQUIRES_OK(context, context->GetAttr("index_name", &index_name));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));
  }

  bool IsExpensive() override
  {
    return true;
  }

  void Compute(OpKernelContext *context) override
  {
    const Tensor &input = context->input(0);
    const TensorShape &input_shape = input.shape();
    const int n = input_shape.dim_size(0);
    const int d = input_shape.dim_size(1);

    IndexContainer *indexContainer = NULL;
    ResourceMgr *rm = context->resource_manager();

    Status s = rm->LookupOrCreate<IndexContainer>(std::string(index_name), std::string("index"), &indexContainer, [d](IndexContainer **ret) {
      *ret = new IndexContainer(d);
      return Status::OK();
    });
    if (s.ok())
    {
      float *flattened_input = new float[d * n];
      auto input_tensor = input.matrix<float>();
      for (int i = 0; i < n; i++)
      {
        for (int j = 0; j < d; j++)
        {
          flattened_input[d * i + j] = input_tensor(i, j);
        }
      }
      if (reset)
      {
        indexContainer->index.reset();
      }
      indexContainer->index.add(n, flattened_input);
      delete[] flattened_input;
    }
    indexContainer->Unref();
    context->SetStatus(s);

    // create output tensor
    Tensor *output = NULL;
    TensorShape output_shape;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    auto output_tensor = output->scalar<int>();
    output_tensor(0) = n;
  }

private:
  bool reset;
  string index_name;
};

REGISTER_OP("AddToIndex")
    .Attr("index_name: string")
    .Attr("reset: bool")
    .Input("to_add: float32")
    .Output("added: float32")
    .SetShapeFn(::tensorflow::shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("AddToIndex").Device(DEVICE_CPU), AddToIndexOp);