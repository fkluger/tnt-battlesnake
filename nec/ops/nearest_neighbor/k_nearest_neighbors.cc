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

class NearestNeighborsOp : public OpKernel
{
  public:
    explicit NearestNeighborsOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("index_name", &index_name));
        OP_REQUIRES_OK(context, context->GetAttr("k", &k));
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

        long *I = new long[k * n];
        float *D = new float[k * n];

        IndexContainer *indexContainer = NULL;
        ResourceMgr *rm = context->resource_manager();

        Status s = rm->LookupOrCreate<IndexContainer>(std::string(index_name), std::string("index"), &indexContainer, [d](IndexContainer **ret) {
            *ret = new IndexContainer(d);
            return Status::OK();
        });
        if (s.ok())
        {
            float *flattened_query = new float[d * n];
            auto input_tensor = input.matrix<float>();
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    flattened_query[d * i + j] = input_tensor(i, j);
                }
            }
            indexContainer->index.search(n, flattened_query, k, D, I);
            delete[] flattened_query;
        }
        indexContainer->Unref();
        context->SetStatus(s);

        // create output tensor
        Tensor *distances_output = NULL;
        Tensor *indices_output = NULL;

        TensorShape output_shape;
        output_shape.AddDim(n);
        output_shape.AddDim(k);

        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &distances_output));
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &indices_output));

        auto distances_output_tensor = distances_output->matrix<float>();
        auto indices_output_tensor = indices_output->matrix<::tensorflow::int64>();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                distances_output_tensor(i, j) = 0.5 * D[i * k + j];
                indices_output_tensor(i, j) = I[i * k + j];
            }
        }
        delete[] I;
        delete[] D;
    }

  private:
    int k;
    string index_name;
};

REGISTER_OP("NearestNeighbors")
    .Attr("index_name: string")
    .Attr("k: int")
    .Input("query: float32")
    .Output("distances: float32")
    .Output("indices: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        int k;
        TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
        c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), k));
        c->set_output(1, c->Matrix(c->Dim(c->input(0), 0), k));
        return Status::OK();
    });
REGISTER_KERNEL_BUILDER(Name("NearestNeighbors").Device(DEVICE_CPU), NearestNeighborsOp);