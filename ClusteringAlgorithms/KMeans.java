package edu.neu.dm.hw3;

import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

class Centroid {
	public double x;
	public double y;
	
	Centroid(double x, double y) {
		this.x = x;
		this.y = y;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(x);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(y);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Centroid other = (Centroid) obj;
		if (Double.doubleToLongBits(x) != Double.doubleToLongBits(other.x))
			return false;
		if (Double.doubleToLongBits(y) != Double.doubleToLongBits(other.y))
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "Centroid [x=" + x + ", y=" + y + "]";
	}
	 
}

public class KMeans {
	static int K = 0;
	static int dataset =0;

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		//For Dataset1
		int seed = 71;
		
		List<DataPoints> dataSet = readDataSet("data/dataset1.txt");
		K = DataPoints.getNoOFLabels(dataSet);
		Collections.shuffle(dataSet, new Random(seed));
		kmeans(dataSet);
		
		System.out.println();
		dataSet = readDataSet("data/dataset2.txt");
		K = DataPoints.getNoOFLabels(dataSet);
		Collections.shuffle(dataSet, new Random(seed));
		kmeans(dataSet);
		
		System.out.println();
		dataSet = readDataSet("data/dataset3.txt");
		K = DataPoints.getNoOFLabels(dataSet);
		Collections.shuffle(dataSet, new Random(seed));
		kmeans(dataSet);
	}
	
	private static void kmeans(List<DataPoints> dataSet) throws IOException {
		
		List<Set<DataPoints>> clusters = new ArrayList<Set<DataPoints>>();
		int k =0;
		while(k < K) {
			Set<DataPoints> cluster = new HashSet<DataPoints>();
			clusters.add(cluster);
			k++;
		}
		
		//Initially randomly assign points to clusters
		int i = 0;
		for(DataPoints point : dataSet) {
			clusters.get(i%K).add(point);
			i++;
		}
		
		//calculate centroid for clusters
		Centroid[] centroids = new Centroid[K];
		for(int j=0; j< K; j++) {
			centroids[j] = getCentroid(clusters.get(j));
			//System.out.println(centroids[j]);
		}

		for(int j=0; j< K; j++) {
			clusters.get(j).removeAll(clusters.get(j));
		}
		
		reassignClusters(dataSet, centroids, clusters);
		
		//continue till converge
		int iteration = 0;
		while(true) {
			iteration++;
			//calculate centroid for clusters
			Centroid[] centroidsNew = new Centroid[K];
			for(int j=0; j< K; j++) {
				centroidsNew[j] = getCentroid(clusters.get(j));
			}

			boolean isConverge = false;
			for(int j=0; j< K; j++) {
				if(!centroidsNew[j].equals(centroids[j])) {
					isConverge = false;
				} else {
					isConverge = true;
				}
			}
			
			if(isConverge)
				break;
			for(int j=0; j< K; j++) {
				clusters.get(j).removeAll(clusters.get(j));
			}
			
			reassignClusters(dataSet, centroidsNew, clusters);
			for(int j=0; j< K; j++) {
				centroids[j] = centroidsNew[j];
			}
		}
		
		System.out.println("Iteration :"+ iteration);
		//Calculate purity
		int[] maxLabelCluster = new int[K]; 
		for(int j=0; j< K; j++) {
			maxLabelCluster[j] = getMaxClusterLabel(clusters.get(j));
		}
		double purity = 0d;
		for(int j=0; j< K; j++) {
			purity += maxLabelCluster[j];
		}
		purity = purity/dataSet.size();
		System.out.println("Purity is :"+purity);
		
		int noOfLabels = DataPoints.getNoOFLabels(dataSet);
		double[][] nmiMatrix = DataPoints.getNMIMatrix(clusters, noOfLabels);
		double nmi = DataPoints.calcNMI(nmiMatrix);
		System.out.println("NMI :"+nmi);
		
		//write clusters to file for plotting
		String filename = "kmeans_dataset"+ ++dataset +".csv";
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("output/"+filename)));
		for(int w=0; w < K; w++) {
			System.out.println("Cluster "+w + " size :"+ clusters.get(w).size());
			//bw.write("\nFor Cluster "+ (w+1) + "\n");
			for(DataPoints point : clusters.get(w))
				bw.write(point.x + "," + point.y + "\n");
			bw.write("\n\n");
		}
		bw.close();
	}
	
	public static int getMaxClusterLabel(Set<DataPoints> cluster) {
		Map<Integer, Integer> labelCounts = new HashMap<Integer, Integer>();
		for(DataPoints point : cluster) {
			if(!labelCounts.containsKey(point.label)) {
				labelCounts.put(point.label, 1);
			} else {
				labelCounts.put(point.label, labelCounts.get(point.label) + 1);
			}
		}
		int max = Integer.MIN_VALUE;
		for(int label : labelCounts.keySet()) {
			if(max < labelCounts.get(label)) 
				max = labelCounts.get(label);
		}
		return max;
	}
	
	private static void reassignClusters(List<DataPoints> dataSet, Centroid[] c, List<Set<DataPoints>> clusters) {
		// reassign points based on cluster and continue till stable clusters found
		double[] dist = new double[K];
		
		for (DataPoints point : dataSet) {

			for(int i=0; i < K; i++) {
				dist[i] = getEuclideanDist(point.x, point.y, c[i].x, c[i].y);
			}
			
			double minDist = getMin(dist);
			
			//assign point to the closest cluster
			/****************Please Fill Missing Lines Here*****************/

			int minIndex = getMinIndex(dist);
			clusters.get(minIndex).add(point);

		}
	}
	
	private static double getMin(double[] dist) {
		// TODO Auto-generated method stub
		double min = Integer.MAX_VALUE;
		for(int i=0; i<dist.length; i++) {
			if(dist[i] < min)
				min = dist[i];
		}
		return min;
	}

	private static int getMinIndex(double[] dist) {
		// TODO Auto-generated method stub
		double min = Integer.MAX_VALUE;
		int index=0;
		for(int i=0; i<dist.length; i++) {
			if(dist[i] < min) {
				index=i;
				min = dist[i];
			}
		}
		return index;
	}


	private static double getEuclideanDist(double x1, double y1, double x2, double y2) {
		double dist = Math.sqrt(Math.pow((x2-x1),2) + Math.pow((y2-y1), 2));
		return dist;
	}
	
	private static Centroid getCentroid(Set<DataPoints> cluster) {
		// TODO Auto-generated method stub
		Double cx=0.0d,cy=0.0d; //mean of x and mean of y
		double size = cluster.size();
		/****************Please Fill Missing Lines Here*****************/
		if(size ==0)
			return new Centroid(0, 0);
		//System.out.println("size is:"+ size );
		for(DataPoints point : cluster) {
				cx += point.x;
				cy += point.y;
		}
		//System.out.println("size is:"+ size );
		//System.out.println("centroids"+ cx +"   "+cy );

		cx = cx/size;
		cy = cy/size;
		return new Centroid(cx, cy);

	}

	public static List<DataPoints> readDataSet(String filePath) {
		List<DataPoints> dataSet = new ArrayList<DataPoints>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(filePath)));
			String line = null;
			while((line = br.readLine()) != null) {
				String[] points = line.split("\t");
				Double x = Double.parseDouble(points[0]);
				Double y = Double.parseDouble(points[1]);
				int label = Integer.parseInt(points[2]);
				DataPoints point = new DataPoints(x,y,label);
				dataSet.add(point);
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		} 
		return dataSet;
	}

}
