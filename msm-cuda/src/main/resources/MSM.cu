#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
//#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define MAX_STR_LEN 256

struct ponto_capturado{
    int TID;
    char *clazz;
    int time;
    double lat, lon;
    int gid;
    int stopId;
};
struct trajetoria{
	ponto_capturado** pontos;
	int qntdPontos;
};

trajetoria** trajetorias;

trajetoria* readTrajFile(char*);

double* trajectoryRawer(trajetoria*);

double euclidean(double *p1, double *p2);

void msm(double* trajA, int lengthA, double* trajB, int lengthB, double* aScore, double* bScore, double* semanticsDescriptors);

double distance(double*, int, double*, int);

int main(int argc, char *argv[]) {
	int file_count = 0;
	int len;
	DIR * dirp;
	struct dirent * entry;

	dirp = opendir("./trajetorias");
	while ((entry = readdir(dirp)) != NULL) {
        len = strlen (entry->d_name);
		if (entry->d_type == DT_REG && strcmp (".traj", &(entry->d_name[len - 5])) == 0) { /* If the entry is a regular file */
			 file_count++;
		}
	}
	closedir(dirp);
	trajetorias  = (trajetoria**) malloc(file_count*sizeof(trajetoria*));
    DIR* FD;
    struct dirent* in_file;
    if (NULL == (FD = opendir ("./trajetorias"))) {
        fprintf(stderr, "Error : Failed to open input directory\n");
        return 1;
    }
	int fileCounter = 0;
    while ((in_file = readdir(FD))) {
        len = strlen (in_file->d_name);
		if (len > 4 && in_file->d_type == DT_REG && strcmp (".traj", &(in_file->d_name[len - 5])) == 0) {
			if (!strcmp (in_file->d_name, "."))
				continue;
			if (!strcmp (in_file->d_name, ".."))    
				continue;
			char filePath[1024];
			sprintf( filePath, "%s/%s", "./trajetorias", in_file->d_name );
			trajetorias[fileCounter++] = readTrajFile(filePath);
		}
	}
	printf("Qntd arquivos lidos %d\n", file_count);
	
	double** allDistances = (double**) malloc(file_count*sizeof(double*));
	double** rawTrajs = (double**) malloc(file_count*sizeof(double*));
	for(int k = 0;k<file_count;k++) {
		rawTrajs[k] = trajectoryRawer(trajetorias[k]);
	}
	for(int k = 0;k<file_count;k++) {
		allDistances[k] = (double*) malloc(file_count*sizeof(double));
	}
	printf("Trajetorias transformadas %d\n", file_count);
	for(int k = 0;k<file_count;k++) {
		allDistances[k][k] = 0.0;
		for(int l = 0;l<file_count;l++) {
	//printf("Distance lengthA=%d, lengthB=%d\n", trajetorias[k]->qntdPontos, trajetorias[l]->qntdPontos);
			if(k<l) {
				double *trajA = rawTrajs[k];
				double *trajB = rawTrajs[l];
				double similarity = distance(trajA, trajetorias[k]->qntdPontos, trajB, trajetorias[l]->qntdPontos);
				allDistances[k][l] = similarity;
				allDistances[l][k] = similarity;
				//printf("Similaridade das trajetórias: %.2f\n", similarity);
			}
		}
	}

	for(int i = 0; i < file_count;i++) {
		if(trajetorias[i]) {
			for(int j = 0; j < trajetorias[i]->qntdPontos;j++) {
				free(trajetorias[i]->pontos[j]);
			}
			free(trajetorias[i]);
		}
	}
	free(trajetorias);
	
	return 0;
}

double distance(double* trajA, int N, double* trajB, int M) {
	double* aScore = (double*)malloc( N*sizeof(double));
	double* bScore = (double*)malloc( N*M*sizeof(double));
	double* semanticsDescriptors = (double*)malloc( 2*2*sizeof(double));
	//GEO
	semanticsDescriptors[0] = 0.0;
	semanticsDescriptors[1] = 0.5;
	//TIME
	semanticsDescriptors[2] = 0.0;
	semanticsDescriptors[3] = 0.5;

	//printf("Distance lengthA=%d, lengthB=%d\n", N,M);
    msm( trajA, N, trajB, M, aScore, bScore, semanticsDescriptors );
    
	double parityAB = 0.0;
	for (int i = 0; i < N; i++) {
		parityAB += aScore[i];
	}

	double parityBA = 0.0;
	for (int i = 0; i < N; i++) {
		double maxScore = 0.0;
		for (int j = 0; j < M; j++) {
			maxScore = MAX(maxScore, bScore[i * M + j]);
		}
		parityBA += maxScore;
	}
	//printf("parityAB=%.2f, parityBA=%.2f\n", parityAB, parityBA );
	double similarity = (parityAB + parityBA) / (N + M);
	free(semanticsDescriptors);
	//printf("similarity=%.2f\n", similarity );
	free(bScore);
	free(aScore);
	aScore = NULL;
	bScore = NULL;
	semanticsDescriptors = NULL;
	
	return similarity;
}

void msm(double* trajA, int lengthA, double* trajB, int lengthB, double* aScore, double* bScore, double* semanticsDescriptors) {
	for(int i = 0; i < lengthA; i++) {
		double latGeoA = trajA[i * 4];
		double lonGeoA = trajA[i * 4 + 1];
		double startTimeA = trajA[i * 4 + 2];
		double endTimeA = trajA[i * 4 + 3];
		
		double geoThreshold = semanticsDescriptors[0];
		double timeThreshold = semanticsDescriptors[2];
	
		double geoWeight = semanticsDescriptors[1];
		double timeWeight = semanticsDescriptors[3];
	
		double maxScore = 0.0;
		for (int j = 0; j < lengthB; j++) {
			double latGeoB = trajB[j * 4];
			double lonGeoB = trajB[j * 4 + 1];
			double startTimeB = trajB[j * 4 + 2];
			double endTimeB = trajB[j * 4 + 3];
			double timeScore = 0.0;
			if(startTimeA < endTimeB && startTimeB < endTimeA ) {
			    double overlap = MIN(endTimeA, endTimeB) - MAX(startTimeA, startTimeB);
			    if(overlap > 0.0) {
	    			double duration = MAX(endTimeA, endTimeB) - MIN(startTimeA, startTimeB);
	    			double timeDistance = 1 - (overlap / duration);
	    			timeScore = (timeDistance <= timeThreshold ? 1 : 0) * timeWeight;
			    }
			}
			double geoB[] = {latGeoB, lonGeoB};
			double geoA[] = {latGeoA, lonGeoA};
			double geoScore = (euclidean(geoB, geoA) <= geoThreshold ? 1 : 0) * geoWeight;
			double sumScore = timeScore + geoScore;
			if(sumScore > maxScore) {
			    maxScore = sumScore;
			}
		    bScore[i * lengthB + j] = sumScore;
		}
		aScore[i] = maxScore;
	}
}

trajetoria* readTrajFile(char *filePath) {
    /* FileStream for the Library File */
    FILE *trajFile;

    /* allocation of the buffer for every line in the File */
    char *buf = (char*) malloc(MAX_STR_LEN);
    char *tmp; 

    /* if the space could not be allocaed, return an error */
    if (buf == NULL) {
        printf ("No memory\n");
        return NULL;
    }

    if ( ( trajFile = fopen( filePath, "r" ) ) == NULL ) //Reading a file
    {
        printf( "File could not be opened: %s.\n", filePath );
		return NULL;
    }
	int pointsCounter = 0;
    while (fgets(buf, MAX_STR_LEN - 1, trajFile) != NULL) {	
		pointsCounter++;
	}
    fclose(trajFile);
	ponto_capturado **traj = (ponto_capturado**) malloc(pointsCounter*sizeof(ponto_capturado*));
	trajetoria* trajetoria = new struct trajetoria;
	trajetoria->pontos = traj;
	trajetoria->qntdPontos = pointsCounter;

    if ( ( trajFile = fopen( filePath, "r" ) ) == NULL ) {
        printf( "File could not be opened: %s.\n", filePath );
		return NULL;
    }
    int i = 0;
    while (fgets(buf, MAX_STR_LEN - 1, trajFile) != NULL)
    {	

        if (strlen(buf)>0) {
	      if(buf[strlen (buf) - 1] == '\n')
	            buf[strlen (buf) - 1] = '\0';
		} else {
			if(buf[0] == '\n') {
				continue;
			}
		}

        tmp = strtok(buf, ";");
		
		traj[i] = new ponto_capturado();
		
        traj[i]->TID = atoi(tmp);

        tmp = strtok(NULL, ";");
		int len = strlen(tmp);
		traj[i]->clazz = (char*)malloc(len + 1);
		strcpy(traj[i]->clazz, tmp);

        tmp = strtok(NULL, ";");
        traj[i]->time = atoi(tmp);

        tmp = strtok(NULL, ";");
        traj[i]->lat = atof(tmp);

        tmp = strtok(NULL, ";");
        traj[i]->lon = atof(tmp);

        tmp = strtok(NULL, ";");
        traj[i]->gid = atoi(tmp);

        tmp = strtok(NULL, ";");

        if ((tmp != NULL) && (tmp[0] == '\0')) {
	        traj[i]->stopId = atoi(tmp);
        } else {
	        traj[i]->stopId = 0;
        }

		/*
        printf("index i= %d  ID: %d, %s, %d, %.8f, %.8f, %d, %d \n",i, traj[i]->TID ,
        					 traj[i]->clazz, traj[i]->time , 
        					 traj[i]->lat, traj[i]->lon,
        					 traj[i]->gid, traj[i]->stopId);
		*/
        i++;
    }
	//printf("Loaded %s - %d points\n", filePath, i);
    fclose(trajFile);
    return trajetoria;
}

double* trajectoryRawer(trajetoria* trajetoria) {
	int N = trajetoria->qntdPontos;
	double* trajA = (double*)malloc( 4*N*sizeof(double));
	for(int i = 0; i < N; i++) {
		trajA[i * 4] = trajetoria->pontos[i]->lat;
		trajA[i * 4 + 1] = trajetoria->pontos[i]->lon;
		trajA[i * 4 + 2] = trajetoria->pontos[i]->time;
		trajA[i * 4 + 3] = trajetoria->pontos[i]->time + 30;
	}
	return trajA;
}

double euclidean(double *p1, double *p2) {
	double distX = abs(p1[0] - p2[0]);
	double distXSquare = distX * distX;

	double distY = abs(p1[1] - p2[1]);
	double distYSquare = distY * distY;

	return sqrt(distXSquare + distYSquare);
}