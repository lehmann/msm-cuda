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
import static jcuda.driver.JCudaDriver.setExceptionsEnabled;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import br.ufsc.core.trajectory.Semantic;
import br.ufsc.core.trajectory.SemanticTrajectory;
import br.ufsc.core.trajectory.TPoint;
import br.ufsc.core.trajectory.TemporalDuration;
import br.ufsc.core.trajectory.Trajectory;
import br.ufsc.ftsm.related.DTW;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class MSMCuda {

	private String ptxFileName;
	private CUdevice device;
	
	private double geoThreshold;
	private double timeThreshold;
	private double geoWeight = 0.5;
	private double timeWeight = 0.5;

	public static void main(String[] args) throws IOException {
//		List<SemanticTrajectory> trajectories = new PatelVehiclesDataReader().read();
		SemanticTrajectory t1 = new SemanticTrajectory(1, 3);
		Instant now = Instant.now();
		for (int i = 0; i < 10; i++) {
			t1.addData(i, Semantic.GEOGRAPHIC, new TPoint(1, 1));
			t1.addData(i, Semantic.TEMPORAL, new TemporalDuration(now.minus(2, ChronoUnit.MINUTES), now));
		}
		SemanticTrajectory t2 = new SemanticTrajectory(1, 3);
		for (int i = 0; i < 10; i++) {
			t2.addData(i, Semantic.GEOGRAPHIC, new TPoint(1, 1));
			t2.addData(i, Semantic.TEMPORAL, new TemporalDuration(now.minus(2, ChronoUnit.MINUTES), now));
		}

		long start = System.nanoTime();
		MSMCuda msmCuda = new MSMCuda();
		msmCuda.init();
		System.out.println(msmCuda.getSimilarity(t1, t2));
		System.out.println("CUDA's time: " + TimeUnit.NANOSECONDS.toMillis((System.nanoTime() - start)));
//		ExecutionStats executionStats = msmCuda.getDistanceStats(t1, t1, 1.0);
//		System.out.println(executionStats.getResult());
//		System.out.println("CUDA's kernel execution time: " + TimeUnit.NANOSECONDS.toMillis(executionStats.getTimeComputing()));
//		start = System.nanoTime();
//		System.out.println(new DTW().getDistance(t1, t1));
//		System.out.println("Java's time: " + TimeUnit.NANOSECONDS.toMillis((System.nanoTime() - start)));
//		
//		DescriptiveStatistics CUDAsKernelStats = new DescriptiveStatistics();
//		DescriptiveStatistics CUDAsStats = new DescriptiveStatistics();
//		DescriptiveStatistics JavaStats = new DescriptiveStatistics();
//		for (int j = 0; j < 40; j++) {
//			start = System.nanoTime();
//			msmCuda.getDistance(t1, t1);
//			CUDAsStats.addValue(System.nanoTime() - start);
//		}
//		for (int j = 0; j < 40; j++) {
//			executionStats = msmCuda.getDistanceStats(t1, t1, 1.0);
//			CUDAsKernelStats.addValue(executionStats.getTimeComputing());
//		}
//		for (int j = 0; j < 40; j++) {
//			start = System.nanoTime();
//			new DTW().getDistance(t1, t1);
//			JavaStats.addValue(System.nanoTime() - start);
//		}
//		System.out.println("CUDAs stats:");
//		printStats(CUDAsStats);
//		System.out.println("CUDAs kernel stats:");
//		printStats(CUDAsKernelStats);
//		System.out.println("Javas stats:");
//		printStats(JavaStats);
	}

	private static void printStats(DescriptiveStatistics stats) {
		System.out.println("Mean - " + TimeUnit.NANOSECONDS.toMillis((long) stats.getMean()));
		System.out.println("Median - " + TimeUnit.NANOSECONDS.toMillis((long) stats.getPercentile(50)));
		System.out.println("Standard deviation - " + TimeUnit.NANOSECONDS.toMillis((long) stats.getStandardDeviation()));
	}

	public void init() throws IOException {
		// Enable exceptions and omit all subsequent error checks
		JCudaDriver.setExceptionsEnabled(true);

		// Create the PTX file by calling the NVCC
		ptxFileName = preparePtxFile("C:/Users/André/workspace/dtw-cuda/src/main/resources/MSMCuda.cu");

		// Initialize the driver and create a context for the first device.
		cuInit(0);
		device = new CUdevice();
		cuDeviceGet(device, 0);
	}
	
//	public ExecutionStats getDistanceStats(Trajectory A, Trajectory B, double warping) throws IOException {
//		double[][] p, q;
//		if (A.length() >= B.length()) {
//			p = toTrajectoryArray(A);
//			q = toTrajectoryArray(B);
//		} else {
//			p = toTrajectoryArray(B);
//			q = toTrajectoryArray(A);
//		}
//
//		CUcontext context = new CUcontext();
//		cuCtxCreate(context, 0, device);
//
//		// Load the ptx file.
//		CUmodule module = new CUmodule();
//		cuModuleLoad(module, ptxFileName);
//		// Obtain a function pointer to the "add" function.
//		CUfunction function = new CUfunction();
//		cuModuleGetFunction(function, module, "dtw");
//
//		// Allocate arrays on the device, one for each row. The pointers
//		// to these array are stored in host memory.
//		CUdeviceptr A_Pointers[] = new CUdeviceptr[p.length];
//		for (int i = 0; i < p.length; i++) {
//			A_Pointers[i] = new CUdeviceptr();
//			cuMemAlloc(A_Pointers[i], 2 * Sizeof.DOUBLE);
//		}
//		// Copy the contents of the rows from the host input data to
//		// the device arrays that have just been allocated.
//		for (int i = 0; i < p.length; i++) {
//			cuMemcpyHtoD(A_Pointers[i], Pointer.to(p[i]), 2 * Sizeof.DOUBLE);
//		}
//
//		// Allocate device memory for the array pointers, and copy
//		// the array pointers from the host to the device.
//		CUdeviceptr deviceInput = new CUdeviceptr();
//		cuMemAlloc(deviceInput, q.length * Sizeof.POINTER);
//		cuMemcpyHtoD(deviceInput, Pointer.to(A_Pointers), q.length * Sizeof.POINTER);
//
//		// Allocate arrays on the device, one for each row. The pointers
//		// to these array are stored in host memory.
//		CUdeviceptr B_Pointers[] = new CUdeviceptr[q.length];
//		for (int i = 0; i < q.length; i++) {
//			B_Pointers[i] = new CUdeviceptr();
//			cuMemAlloc(B_Pointers[i], 2 * Sizeof.DOUBLE);
//		}
//		// Copy the contents of the rows from the host input data to
//		// the device arrays that have just been allocated.
//		for (int i = 0; i < q.length; i++) {
//			cuMemcpyHtoD(B_Pointers[i], Pointer.to(q[i]), 2 * Sizeof.DOUBLE);
//		}
//
//		// Allocate device memory for the array pointers, and copy
//		// the array pointers from the host to the device.
//		CUdeviceptr deviceInputB = new CUdeviceptr();
//		cuMemAlloc(deviceInputB, q.length * Sizeof.POINTER);
//		cuMemcpyHtoD(deviceInputB, Pointer.to(B_Pointers), q.length * Sizeof.POINTER);
//
//		// Allocate device output memory: A single column with
//		// height 'numThreads'.
//		CUdeviceptr deviceOutput = new CUdeviceptr();
//		cuMemAlloc(deviceOutput, Sizeof.FLOAT);
//
//        // Set up the kernel parameters: A pointer to an array
//        // of pointers which point to the actual values.
//        Pointer kernelParams = Pointer.to(
//            Pointer.to(deviceInput), 
//            Pointer.to(new int[]{p.length}), 
//            Pointer.to(deviceInputB), 
//            Pointer.to(new int[]{q.length}), 
//            Pointer.to(new double[]{warping}), 
//            Pointer.to(deviceOutput)
//        );
//        
//        // Call the kernel function.
//
//		long start = System.nanoTime();
//        cuLaunchKernel(function, 
//            1, 1, 1,           // Grid dimension 
//            p.length, 1, 1,  // Block dimension
//            0, null,           // Shared memory size and stream 
//            kernelParams, null // Kernel- and extra parameters
//        ); 
//		cuCtxSynchronize();
//		long end = System.nanoTime() - start;
//
//		// Allocate host output memory and copy the device output
//		// to the host.
//		float hostOutput[] = new float[1];
//		cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, Sizeof.FLOAT);
//
//		// Clean up.
//		cuMemFree(deviceInput);
//		cuMemFree(deviceInputB);
//		cuMemFree(deviceOutput);
//		
//		cuCtxDestroy(context);
//
//		return new ExecutionStats(hostOutput[0], end);
//	}
	
	public double getSimilarity(SemanticTrajectory A, SemanticTrajectory B) throws IOException {
		double[][] trajA = toTrajectoryArray(A);
		double[][] trajB = toTrajectoryArray(B);

		// Enable exceptions and omit all subsequent error checks
		JCudaDriver.setExceptionsEnabled(true);

		// Create the PTX file by calling the NVCC
		String ptxFileName = preparePtxFile("C:/Users/André/workspace/dtw-cuda/src/main/resources/MSMCuda.cu");

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
		cuModuleGetFunction(function, module, "msm");

		// Allocate arrays on the device, one for each row. The pointers
		// to these array are stored in host memory.
		CUdeviceptr A_Pointers[] = new CUdeviceptr[trajA.length];
		for (int i = 0; i < trajA.length; i++) {
			A_Pointers[i] = new CUdeviceptr();
			cuMemAlloc(A_Pointers[i], 4 * Sizeof.DOUBLE);
		}
		// Copy the contents of the rows from the host input data to
		// the device arrays that have just been allocated.
		for (int i = 0; i < trajA.length; i++) {
			cuMemcpyHtoD(A_Pointers[i], Pointer.to(trajA[i]), 4 * Sizeof.DOUBLE);
		}

		// Allocate device memory for the array pointers, and copy
		// the array pointers from the host to the device.
		CUdeviceptr deviceTrajA = new CUdeviceptr();
		cuMemAlloc(deviceTrajA, trajA.length * Sizeof.POINTER);
		cuMemcpyHtoD(deviceTrajA, Pointer.to(A_Pointers), trajA.length * Sizeof.POINTER);

		// Allocate arrays on the device, one for each row. The pointers
		// to these array are stored in host memory.
		CUdeviceptr B_Pointers[] = new CUdeviceptr[trajB.length];
		for (int i = 0; i < trajB.length; i++) {
			B_Pointers[i] = new CUdeviceptr();
			cuMemAlloc(B_Pointers[i], 4 * Sizeof.DOUBLE);
		}
		// Copy the contents of the rows from the host input data to
		// the device arrays that have just been allocated.
		for (int i = 0; i < trajB.length; i++) {
			cuMemcpyHtoD(B_Pointers[i], Pointer.to(trajB[i]), 4 * Sizeof.DOUBLE);
		}

		// Allocate device memory for the array pointers, and copy
		// the array pointers from the host to the device.
		CUdeviceptr deviceTrajB = new CUdeviceptr();
		cuMemAlloc(deviceTrajB, trajB.length * Sizeof.POINTER);
		cuMemcpyHtoD(deviceTrajB, Pointer.to(B_Pointers), trajB.length * Sizeof.POINTER);

		// Allocate device output memory: A single column with
		// height 'numThreads'.
		CUdeviceptr aScorePtr = new CUdeviceptr();
		cuMemAlloc(aScorePtr, trajA.length * Sizeof.DOUBLE);
		CUdeviceptr bScorePtr = new CUdeviceptr();
		cuMemAlloc(bScorePtr, trajA.length * trajB.length * Sizeof.DOUBLE);
		
		CUdeviceptr sem_descriptors[] = new CUdeviceptr[2];
		for (int i = 0; i < 2; i++) {
			sem_descriptors[i] = new CUdeviceptr();
			cuMemAlloc(sem_descriptors[i], 2 * Sizeof.DOUBLE);
		}
		cuMemcpyHtoD(sem_descriptors[0], Pointer.to(new double[] {geoThreshold, geoWeight}), 2 * Sizeof.DOUBLE);
		cuMemcpyHtoD(sem_descriptors[1], Pointer.to(new double[] {timeThreshold, timeWeight}), 2 * Sizeof.DOUBLE);
		
		CUdeviceptr semanticsDescriptors = new CUdeviceptr();
		cuMemAlloc(semanticsDescriptors, trajB.length * Sizeof.POINTER);
		cuMemcpyHtoD(semanticsDescriptors, Pointer.to(sem_descriptors), 2 * Sizeof.POINTER);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParams = Pointer.to(
            Pointer.to(deviceTrajA), 
            Pointer.to(new int[]{trajA.length}), 
            Pointer.to(deviceTrajB), 
            Pointer.to(new int[]{trajB.length}), 
            Pointer.to(aScorePtr),  
            Pointer.to(bScorePtr), 
            Pointer.to(semanticsDescriptors)
        );
        
        int THREADS = 128;
        int BLOCKS = trajB.length / THREADS + 1;
        
//        setExceptionsEnabled(false);
        
        // Call the kernel function.
        cuLaunchKernel(function, 
            THREADS, 1, 1,           // Grid dimension 
            BLOCKS, 1, 1,  // Block dimension
            0, null,           // Shared memory size and stream 
            kernelParams, null // Kernel- and extra parameters
        ); 
		cuCtxSynchronize();

		// Allocate host output memory and copy the device output
		// to the host.
		double aScore[] = new double[trajA.length];
		cuMemcpyDtoH(Pointer.to(aScore), aScorePtr, trajA.length * Sizeof.DOUBLE);
		double bScore[] = new double[trajA.length * trajB.length];
		cuMemcpyDtoH(Pointer.to(bScore), bScorePtr, trajA.length * trajB.length * Sizeof.DOUBLE);

		// Clean up.
		cuMemFree(deviceTrajA);
		cuMemFree(deviceTrajB);
		cuMemFree(aScorePtr);
		cuMemFree(bScorePtr);
		cuMemFree(semanticsDescriptors);
		
		cuCtxDestroy(context);

		double parityAB = 0.0;
		for (int i = 0; i < aScore.length; i++) {
			parityAB += aScore[i];
		}
		double parityBA = 0.0;
		for (int i = 0; i < bScore.length; i+= trajA.length) {
			double maxScore = 0.0;
			for (int j = 0; j < trajB.length; j++) {
				maxScore = Math.max(maxScore, bScore[i + j]);
			}
			parityBA += maxScore;
		}
		double similarity = (parityAB + parityBA) / (A.length() + B.length());

		return similarity;
	}
	
//	public double _getDistance(Trajectory A, Trajectory B, double warping) {
//		double[][] p, q;
//		int sizeA, sizeB;
//		if (A.length() >= B.length()) {
//			p = toTrajectoryArray(A);
//			q = toTrajectoryArray(B);
//		} else {
//			p = toTrajectoryArray(B);
//			q = toTrajectoryArray(A);
//		}
//		sizeA = p.length;
//		sizeB = q.length;
//
//		return a(warping, p, q, sizeA, sizeB);
//
//	}

	private double a(double warping, double[][] p, double[][] q, int sizeA, int sizeB) {
		// "DTW matrix" in linear space.
		double[][] dtwMatrix = new double[2][sizeA + 1];
		// The absolute size of the warping window (to each side of the main
		// diagonal)
		int w = (int) Math.ceil((sizeA) * warping);

		// Initialization (all elements of the first line are INFINITY, except
		// the 0th, and
		// the same value is given to the first element of the first analyzed
		// line).
		for (int i = 0; i <= sizeA; i++) {
			dtwMatrix[0][i] = Double.POSITIVE_INFINITY;
			dtwMatrix[1][i] = Double.POSITIVE_INFINITY;
		}
		dtwMatrix[0][0] = 0;

		// Distance calculation
		for (int i = 1; i <= sizeB; i++) {

			int beg = Math.max(1, i - w);
			int end = Math.min(i + w, sizeA);

			int thisI = i % 2;
			int prevI = (i - 1) % 2;

			// Fixing values to this iteration
			dtwMatrix[i % 2][beg - 1] = Double.POSITIVE_INFINITY;

			for (int j = beg; j <= end; j++) {

				// DTW(i,j) = c(i-1,j-1) + min(DTW(i-1,j-1), DTW(i,j-1),
				// DTW(i-1,j)).
				dtwMatrix[i % 2][j] = euclidean(q[i - 1], p[j - 1])
						+ Math.min(dtwMatrix[thisI][j - 1], Math.min(dtwMatrix[prevI][j], dtwMatrix[prevI][j - 1]));

			}

		}

		return dtwMatrix[sizeB % 2][sizeA];
	}

	private double[][] toTrajectoryArray(SemanticTrajectory a) {
		double[][] ret = new double[a.length()][4];
		for (int i = 0; i < a.length(); i++) {
			TPoint geo = Semantic.GEOGRAPHIC.getData(a, i);
			ret[i][0] = geo.getX();
			ret[i][1] = geo.getY();
			TemporalDuration time = Semantic.TEMPORAL.getData(a, i);
			ret[i][2] = time.getStart().toEpochMilli();
			ret[i][3] = time.getEnd().toEpochMilli();
		}
		return ret;
	}

	public static double euclidean(double[] p1, double[] p2) {
		double distX = Math.abs(p1[0] - p2[0]);
		double distXSquare = distX * distX;

		double distY = Math.abs(p1[1] - p2[1]);
		double distYSquare = distY * distY;

		return Math.sqrt(distXSquare + distYSquare);
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
		File cuFile = new File(cuFileName);
		if (!cuFile.exists()) {
			throw new IOException("Input file not found: " + cuFileName);
		}
		String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
		File ptxFile = new File(ptxFileName);
		if (ptxFile.exists()) {
			if(ptxFile.lastModified() > cuFile.lastModified()) {
				return ptxFileName;
			}
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
