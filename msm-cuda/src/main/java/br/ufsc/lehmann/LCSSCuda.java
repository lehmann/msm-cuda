package br.ufsc.lehmann;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.sql.Timestamp;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import br.ufsc.core.trajectory.Semantic;
import br.ufsc.core.trajectory.TPoint;
import br.ufsc.core.trajectory.Trajectory;
import br.ufsc.ftsm.base.TrajectorySimilarityCalculator;
import br.ufsc.ftsm.related.DTW;
import br.ufsc.ftsm.related.LCSS;
import br.ufsc.ftsm.related.LCSS.LCSSSemanticParameter;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class LCSSCuda {

	public static void main(String[] args) throws IOException {
		Trajectory t1 = new Trajectory(1);
		for (int i = 0; i < 100; i++) {
			t1.addPoint(1, 1, 1, new Timestamp(8), 0);
			t1.addPoint(1, 2, 2, new Timestamp(9), 0);
			t1.addPoint(1, 3, 3, new Timestamp(10), 0);
			t1.addPoint(1, 4, 4, new Timestamp(11), 0);
			t1.addPoint(1, 5, 5, new Timestamp(12), 0);
			t1.addPoint(1, 6, 6, new Timestamp(13), 0);
		}

		long start = System.nanoTime();
		LCSSCuda dtwCuda = new LCSSCuda();
		System.out.println(dtwCuda.getDistance(t1, t1, 100));
		System.out.println("CUDA's time: " + TimeUnit.NANOSECONDS.toMillis((System.nanoTime() - start)));
		ExecutionStats executionStats = dtwCuda.getDistanceStats(t1, t1, 100);
		System.out.println(executionStats.getResult());
		System.out.println("CUDA's kernel execution time: " + TimeUnit.NANOSECONDS.toMillis(executionStats.getTimeComputing()));
		start = System.nanoTime();
		System.out.println(new LCSS(new LCSSSemanticParameter(Semantic.GEOGRAPHIC, 100)).getDistance(t1, t1));
		System.out.println("Java's time: " + TimeUnit.NANOSECONDS.toMillis((System.nanoTime() - start)));
		
		DescriptiveStatistics CUDAsKernelStats = new DescriptiveStatistics();
		DescriptiveStatistics CUDAsStats = new DescriptiveStatistics();
		DescriptiveStatistics JavaStats = new DescriptiveStatistics();
		for (int j = 0; j < 40; j++) {
			start = System.nanoTime();
			dtwCuda.getDistance(t1, t1);
			CUDAsStats.addValue(System.nanoTime() - start);
		}
		for (int j = 0; j < 40; j++) {
			executionStats = dtwCuda.getDistanceStats(t1, t1, 1.0);
			CUDAsKernelStats.addValue(executionStats.getTimeComputing());
		}
		for (int j = 0; j < 40; j++) {
			start = System.nanoTime();
			new DTW().getDistance(t1, t1);
			JavaStats.addValue(System.nanoTime() - start);
		}
		System.out.println("CUDAs stats:");
		printStats(CUDAsStats);
		System.out.println("CUDAs kernel stats:");
		printStats(CUDAsKernelStats);
		System.out.println("Javas stats:");
		printStats(JavaStats);
	}

	private static void printStats(DescriptiveStatistics stats) {
		System.out.println("Mean - " + TimeUnit.NANOSECONDS.toMillis((long) stats.getMean()));
		System.out.println("Median - " + TimeUnit.NANOSECONDS.toMillis((long) stats.getPercentile(50)));
		System.out.println("Standard deviation - " + TimeUnit.NANOSECONDS.toMillis((long) stats.getStandardDeviation()));
	}
	
	public ExecutionStats getDistanceStats(Trajectory A, Trajectory B, double threshold) throws IOException {
		double[][] p, q;
		if (A.length() >= B.length()) {
			p = toTrajectoryArray(A);
			q = toTrajectoryArray(B);
		} else {
			p = toTrajectoryArray(B);
			q = toTrajectoryArray(A);
		}
		// Enable exceptions and omit all subsequent error checks
		JCudaDriver.setExceptionsEnabled(true);

		// Create the PTX file by calling the NVCC
		String ptxFileName = preparePtxFile("C:/Users/André/workspace/dtw-cuda/target/classes/DTWCuda.cu");

		// Initialize the driver and create a context for the first device.
		cuInit(0);
		CUdevice device = new CUdevice();
		cuDeviceGet(device, 0);
		CUcontext context = new CUcontext();
		cuCtxCreate(context, 0, device);

		// Load the ptx file.
		CUmodule module = new CUmodule();
		cuModuleLoad(module, ptxFileName);

		// Obtain a function pointer to the "add" function.
		CUfunction function = new CUfunction();
		cuModuleGetFunction(function, module, "lcss");

		// Allocate arrays on the device, one for each row. The pointers
		// to these array are stored in host memory.
		CUdeviceptr A_Pointers[] = new CUdeviceptr[p.length];
		for (int i = 0; i < p.length; i++) {
			A_Pointers[i] = new CUdeviceptr();
			cuMemAlloc(A_Pointers[i], 2 * Sizeof.DOUBLE);
		}
		// Copy the contents of the rows from the host input data to
		// the device arrays that have just been allocated.
		for (int i = 0; i < p.length; i++) {
			cuMemcpyHtoD(A_Pointers[i], Pointer.to(p[i]), 2 * Sizeof.DOUBLE);
		}

		// Allocate device memory for the array pointers, and copy
		// the array pointers from the host to the device.
		CUdeviceptr deviceInput = new CUdeviceptr();
		cuMemAlloc(deviceInput, q.length * Sizeof.POINTER);
		cuMemcpyHtoD(deviceInput, Pointer.to(A_Pointers), q.length * Sizeof.POINTER);

		// Allocate arrays on the device, one for each row. The pointers
		// to these array are stored in host memory.
		CUdeviceptr B_Pointers[] = new CUdeviceptr[q.length];
		for (int i = 0; i < q.length; i++) {
			B_Pointers[i] = new CUdeviceptr();
			cuMemAlloc(B_Pointers[i], 2 * Sizeof.DOUBLE);
		}
		// Copy the contents of the rows from the host input data to
		// the device arrays that have just been allocated.
		for (int i = 0; i < q.length; i++) {
			cuMemcpyHtoD(B_Pointers[i], Pointer.to(q[i]), 2 * Sizeof.DOUBLE);
		}

		// Allocate device memory for the array pointers, and copy
		// the array pointers from the host to the device.
		CUdeviceptr deviceInputB = new CUdeviceptr();
		cuMemAlloc(deviceInputB, q.length * Sizeof.POINTER);
		cuMemcpyHtoD(deviceInputB, Pointer.to(B_Pointers), q.length * Sizeof.POINTER);

		// Allocate device output memory: A single column with
		// height 'numThreads'.
		CUdeviceptr deviceOutput = new CUdeviceptr();
		cuMemAlloc(deviceOutput, Sizeof.FLOAT);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParams = Pointer.to(
            Pointer.to(deviceInput), 
            Pointer.to(new int[]{p.length}), 
            Pointer.to(deviceInputB), 
            Pointer.to(new int[]{q.length}), 
            Pointer.to(new double[]{threshold}), 
            Pointer.to(deviceOutput)
        );
        
        // Call the kernel function.

		long start = System.nanoTime();
        cuLaunchKernel(function, 
            1, 1, 1,           // Grid dimension 
            p.length, 1, 1,  // Block dimension
            0, null,           // Shared memory size and stream 
            kernelParams, null // Kernel- and extra parameters
        ); 
		cuCtxSynchronize();
		long end = System.nanoTime() - start;

		// Allocate host output memory and copy the device output
		// to the host.
		float hostOutput[] = new float[1];
		cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, Sizeof.FLOAT);

		// Clean up.
		cuMemFree(deviceInput);
		cuMemFree(deviceInputB);
		cuMemFree(deviceOutput);

		cuCtxDestroy(context);
		
		return new ExecutionStats(hostOutput[0], end);
	}
	
	public double getDistance(Trajectory t1, Trajectory t2) {
		try {
			return getDistance(t1, t2, 1.0);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public double getDistance(Trajectory A, Trajectory B, double threshold) throws IOException {
		double[][] p, q;
		if (A.length() >= B.length()) {
			p = toTrajectoryArray(A);
			q = toTrajectoryArray(B);
		} else {
			p = toTrajectoryArray(B);
			q = toTrajectoryArray(A);
		}

		// Enable exceptions and omit all subsequent error checks
		JCudaDriver.setExceptionsEnabled(true);

		// Create the PTX file by calling the NVCC
		String ptxFileName = preparePtxFile("C:/Users/André/workspace/dtw-cuda/target/classes/LCSSCuda.cu");

		// Initialize the driver and create a context for the first device.
		cuInit(0);
		CUdevice device = new CUdevice();
		cuDeviceGet(device, 0);
		CUcontext context = new CUcontext();
		cuCtxCreate(context, 0, device);

		// Load the ptx file.
		CUmodule module = new CUmodule();
		cuModuleLoad(module, ptxFileName);

		// Obtain a function pointer to the "add" function.
		CUfunction function = new CUfunction();
		cuModuleGetFunction(function, module, "lcss");

		// Allocate arrays on the device, one for each row. The pointers
		// to these array are stored in host memory.
		CUdeviceptr A_Pointers[] = new CUdeviceptr[p.length];
		for (int i = 0; i < p.length; i++) {
			A_Pointers[i] = new CUdeviceptr();
			cuMemAlloc(A_Pointers[i], 2 * Sizeof.DOUBLE);
		}
		// Copy the contents of the rows from the host input data to
		// the device arrays that have just been allocated.
		for (int i = 0; i < p.length; i++) {
			cuMemcpyHtoD(A_Pointers[i], Pointer.to(p[i]), 2 * Sizeof.DOUBLE);
		}

		// Allocate device memory for the array pointers, and copy
		// the array pointers from the host to the device.
		CUdeviceptr deviceInput = new CUdeviceptr();
		cuMemAlloc(deviceInput, q.length * Sizeof.POINTER);
		cuMemcpyHtoD(deviceInput, Pointer.to(A_Pointers), q.length * Sizeof.POINTER);

		// Allocate arrays on the device, one for each row. The pointers
		// to these array are stored in host memory.
		CUdeviceptr B_Pointers[] = new CUdeviceptr[q.length];
		for (int i = 0; i < q.length; i++) {
			B_Pointers[i] = new CUdeviceptr();
			cuMemAlloc(B_Pointers[i], 2 * Sizeof.DOUBLE);
		}
		// Copy the contents of the rows from the host input data to
		// the device arrays that have just been allocated.
		for (int i = 0; i < q.length; i++) {
			cuMemcpyHtoD(B_Pointers[i], Pointer.to(q[i]), 2 * Sizeof.DOUBLE);
		}

		// Allocate device memory for the array pointers, and copy
		// the array pointers from the host to the device.
		CUdeviceptr deviceInputB = new CUdeviceptr();
		cuMemAlloc(deviceInputB, q.length * Sizeof.POINTER);
		cuMemcpyHtoD(deviceInputB, Pointer.to(B_Pointers), q.length * Sizeof.POINTER);

		// Allocate device output memory: A single column with
		// height 'numThreads'.
		CUdeviceptr deviceOutput = new CUdeviceptr();
		cuMemAlloc(deviceOutput, Sizeof.FLOAT);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParams = Pointer.to(
            Pointer.to(deviceInput), 
            Pointer.to(new int[]{p.length}), 
            Pointer.to(deviceInputB), 
            Pointer.to(new int[]{q.length}), 
            Pointer.to(new double[]{threshold}), 
            Pointer.to(deviceOutput)
        );
        
        // Call the kernel function.
        cuLaunchKernel(function, 
            1, 1, 1,           // Grid dimension 
            p.length, 1, 1,  // Block dimension
            0, null,           // Shared memory size and stream 
            kernelParams, null // Kernel- and extra parameters
        ); 
		cuCtxSynchronize();

		// Allocate host output memory and copy the device output
		// to the host.
		float hostOutput[] = new float[1];
		cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, Sizeof.FLOAT);

		// Clean up.
		cuMemFree(deviceInput);
		cuMemFree(deviceInputB);
		cuMemFree(deviceOutput);

		cuCtxDestroy(context);

		return hostOutput[0];
	}

	private double[][] toTrajectoryArray(Trajectory a) {
		List<TPoint> points = a.getPoints();
		double[][] ret = new double[points.size()][2];
		for (int i = 0; i < points.size(); i++) {
			ret[i][0] = points.get(i).getX();
			ret[i][1] = points.get(i).getY();
		}
		return ret;
	}

	/**
	 * The extension of the given file name is replaced with "ptx". If the file
	 * with the resulting name does not exist, it is compiled from the given
	 * file using NVCC. The name of the PTX file is returned.
	 *
	 * @param cuFileName
	 *            The name of the .CU file
	 * @return The name of the PTX file
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	private static String preparePtxFile(String cuFileName) throws IOException {
		int endIndex = cuFileName.lastIndexOf('.');
		if (endIndex == -1) {
			endIndex = cuFileName.length() - 1;
		}
		String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
		File ptxFile = new File(ptxFileName);
		if (ptxFile.exists()) {
			return ptxFileName;
		}

		File cuFile = new File(cuFileName);
		if (!cuFile.exists()) {
			throw new IOException("Input file not found: " + cuFileName);
		}
		String modelString = "-m" + System.getProperty("sun.arch.data.model");
		String command = "nvcc " + modelString + " -ptx " + cuFile.getPath() + " -o " + ptxFileName;

		System.out.println("Executing\n" + command);
		Process process = Runtime.getRuntime().exec(command);

		String errorMessage = new String(toByteArray(process.getErrorStream()));
		String outputMessage = new String(toByteArray(process.getInputStream()));
		int exitValue = 0;
		try {
			exitValue = process.waitFor();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new IOException("Interrupted while waiting for nvcc output", e);
		}

		if (exitValue != 0) {
			System.out.println("nvcc process exitValue " + exitValue);
			System.out.println("errorMessage:\n" + errorMessage);
			System.out.println("outputMessage:\n" + outputMessage);
			throw new IOException("Could not create .ptx file: " + errorMessage);
		}

		System.out.println("Finished creating PTX file");
		return ptxFileName;
	}

	/**
	 * Fully reads the given InputStream and returns it as a byte array
	 *
	 * @param inputStream
	 *            The input stream to read
	 * @return The byte array containing the data from the input stream
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	private static byte[] toByteArray(InputStream inputStream) throws IOException {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		byte buffer[] = new byte[8192];
		while (true) {
			int read = inputStream.read(buffer);
			if (read == -1) {
				break;
			}
			baos.write(buffer, 0, read);
		}
		return baos.toByteArray();
	}
}
