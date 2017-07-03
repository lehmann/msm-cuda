package br.ufsc.lehmann;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import br.ufsc.core.trajectory.SemanticTrajectory;

public class PatelVehiclesDataReader {

	public List<SemanticTrajectory> read() throws IOException {
		CSVParser bikeParser = CSVParser.parse(//
				new File("./src/main/resources/backup_patel_vehicles.csv"), Charset.defaultCharset(),//
				CSVFormat.EXCEL.withHeader("tid","class","time","latitude","longitude","gid","semantic_stop_id"));

		List<CSVRecord> vehicles = bikeParser.getRecords();
		return null;
	}
}
