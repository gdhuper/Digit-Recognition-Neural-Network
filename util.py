import pyopencl as cl
import numpy as np

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

#(n, m, p) = (700, 700, 700)

def dotProduct(a, b):
	
	#a = np.random.randint(255, size=(n,m))
	#b = np.random.randint(255, size=(m,p))
	n = a.shape[0]
	m = a.shape[1]
	p = b.shape[1]

	c = np.zeros((n*p), dtype=np.float32)

	a = a.astype(np.float32)

	b = b.astype(np.float32)


	platform = cl.get_platforms() 
	gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)

	#building context using gpu 
	ctx = cl.Context(devices=gpu_devices)

	#initiating queue with context
	queue = cl.CommandQueue(ctx)

	mf = cl.mem_flags

	#transforming input vectors to device memory 
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

	#buffer for resulting vector
	c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)


	prg = cl.Program(ctx, """
	    __kernel void multiply(ushort n,
	    ushort m, ushort p, __global float *a,
	    __global float *b, __global float *c)
	    {
	      int gid = get_global_id(0);
	      c[gid] = 0.0f;
	      int rowC = gid/p;
	      int colC = gid%p;
	      __global float *pA = &a[rowC*m];
	      __global float *pB = &b[colC];
	      for(int k=0; k<m; k++)
	      {
	         pB = &b[colC+k*p];
	         c[gid] += (*(pA++))*(*pB);
	      }
	    }
	    """ ).build()


	#executing dot product of matrix a and matrix b
	prg.multiply(queue, c.shape, None,
	             np.uint16(n), np.uint16(m), np.uint16(p),
	             a_buf, b_buf, c_buf)

	#empty matrix to hold dot product
	a_mul_b = np.empty_like(c)

	#copying result from buffer to result matrix
	cl.enqueue_copy(queue, a_mul_b, c_buf)

	print("matrix A:")
	print(a.reshape(n, m))
	print("matrix B:")
	print(b.reshape(m, p))
	print("multiplied A*B:")
	print(a_mul_b)
	#print(a_mul_b.reshape(n, p))
	return a_mul_b




#helper function to unit test function
def test():
	#nn = NeuralNetwork("inputNodes", "hiddleNodes", "outNodes", "learningRate")

	a = np.random.randint(255, size=(100,784))
	b = np.random.randint(255, size=(784,1))

	start_time = time.time()
	#print(nn.dotproduct(a, b))
	t1 = (time.time() - start_time)
	print("--- %s seconds ---" % t1)

	start_time2 = time.time()
	print(a.shape[0], a.shape[1], b.shape[0], b.shape[1])
	#dotProduct(a, b)
	t2 = (time.time() - start_time2)
	print("--- %s seconds ---" % t2)

	print("Pure Python: ", t1)
	print("OpenCL: ", t2)
	print("Pure Python is faster" if t1 < t2 else "OpenCl is faster")