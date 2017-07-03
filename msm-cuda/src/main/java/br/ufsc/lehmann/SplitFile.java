package br.ufsc.lehmann;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class SplitFile {

	public static void main(String[] args) throws IOException {
		List<String> lines = Files.readAllLines(Paths.get("C:/Users/André/workspace/dtw-cuda/src/main/resources/backup_patel_vehicles.csv"));
		Map<String, List<String>> allTrajectories = new HashMap<String, List<String>>();
		for (String line : lines) {
			String[] fields = line.split(";");
			if(!fields[0].equals("\"tid\"")) {
				String tid = fields[0];
				tid = tid.substring(1, tid.length() - 2).trim();
				String clazz = fields[1];
				clazz = clazz.substring(1, clazz.length() - 2).trim();
				if(!allTrajectories.containsKey(tid)) {
					allTrajectories.put(tid, new ArrayList<String>());
				}
				if(fields.length == 7) {
					allTrajectories.get(tid).add(String.format("%s;%s;%s;%s;%s;%s;%s", tid, clazz, fields[2], fields[3], fields[4], fields[5], fields[6]));
				} else {
					allTrajectories.get(tid).add(String.format("%s;%s;%s;%s;%s;%s;", tid, clazz, fields[2], fields[3], fields[4], fields[5]));
				}
			}
		}
		for (Map.Entry<String, List<String>> entry : allTrajectories.entrySet()) {
			Path file = Files.createFile(Paths.get("C:/Users/André/workspace/dtw-cuda/src/main/resources/" + entry.getKey() + ".traj"));
			FileWriter writer = new FileWriter(file.toFile());
			for (Iterator iterator = entry.getValue().iterator(); iterator.hasNext();) {
				String traj = (String) iterator.next();
				writer.write(traj);
				writer.write("\n");
			}
			writer.flush();
			writer.close();
		}
	}
}
