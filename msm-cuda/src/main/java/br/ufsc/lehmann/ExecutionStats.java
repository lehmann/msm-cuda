package br.ufsc.lehmann;

public class ExecutionStats {
	private double result;
	private long timeComputing;
	public ExecutionStats(double result, long time) {
		this.result = result;
		timeComputing = time;
	}
	public double getResult() {
		return result;
	}
	public void setResult(double result) {
		this.result = result;
	}
	public long getTimeComputing() {
		return timeComputing;
	}
	public void setTimeComputing(long timeComputing) {
		this.timeComputing = timeComputing;
	}
}
