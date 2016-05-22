package edu.neu.dm.hw3;

import java.io.*;
import java.util.*;

public class DataPoints {
	public double x;
	public double y;
	public int label;
	public boolean isAssignedToCluster;
	
	DataPoints(double x, double y, int label) {
		this.x = x;
		this.y = y;
		this.label = label;
		this.isAssignedToCluster = false;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + label;
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
		DataPoints other = (DataPoints) obj;
		if (label != other.label)
			return false;
		if (Double.doubleToLongBits(x) != Double.doubleToLongBits(other.x))
			return false;
		if (Double.doubleToLongBits(y) != Double.doubleToLongBits(other.y))
			return false;
		return true;
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
	
	public static double[][] getNMIMatrix(List<Set<DataPoints>> clusters, int noOfLabels) {
		double[][] nmiMatrix = new double[noOfLabels+1][clusters.size()+1];
		int clusterNo = 0;
		for(Set<DataPoints> cluster : clusters) {
			Map<Integer, Integer> labelCounts = new TreeMap<Integer, Integer>();
			for(DataPoints point : cluster) {
				if(!labelCounts.containsKey(point.label)) {
					labelCounts.put(point.label, 1);
				} else {
					labelCounts.put(point.label, labelCounts.get(point.label) + 1);
				}
			}
			int max = Integer.MIN_VALUE;
			int labelNo =0;
			int labelTotal =0;
			for(int label : labelCounts.keySet()) {
				nmiMatrix[label-1][clusterNo] = labelCounts.get(label);
				labelTotal += labelCounts.get(label);
//				System.out.println(//"Label : value ="+);
			}
			nmiMatrix[noOfLabels][clusterNo] = labelTotal;
			clusterNo++;
			labelCounts.clear();
		}
		
		//populate last col
		int lastRowCol = 0;
		for(int i=0; i< nmiMatrix.length-1; i++) {
			int totalRow = 0;
			for(int j=0; j<nmiMatrix[i].length-1; j++) {
				totalRow += nmiMatrix[i][j];
			}
			lastRowCol += totalRow; 
			nmiMatrix[i][clusters.size()] = totalRow;
		}
		nmiMatrix[noOfLabels][clusters.size()] = lastRowCol;
		
		return 	nmiMatrix;
	}
	
	public static double calcNMI(double[][] nmiMatrix) {
		// TODO Auto-generated method stub
		//calculate I
		int row = nmiMatrix.length;
		int col = nmiMatrix[0].length;
		double N = nmiMatrix[row-1][col-1]; 
				
		double I = 0d;
		double HOmega = 0d;
		double HC = 0d;
		
		for(int i=0; i<row-1; i++) {
			for(int j=0; j<col-1; j++) {
				double logPart = (N*nmiMatrix[i][j])/(nmiMatrix[i][col-1]*nmiMatrix[row-1][j]);
				if(logPart == 0d)
					continue;
				I += (nmiMatrix[i][j]/N)*Math.log(logPart);
				double logPart1 = nmiMatrix[row-1][j]/N;
				if(logPart1 == 0d)
					continue;
				HC += nmiMatrix[row-1][j]/N * Math.log(logPart1); 
			}
			HOmega += nmiMatrix[i][col-1]/N * Math.log(nmiMatrix[i][col-1]/N);
		}
		
		return I/Math.sqrt(HC * HOmega);
	}
	
	public static int getNoOFLabels(List<DataPoints> dataSet) {
		Set<Integer> labels = new HashSet<Integer>();
		for(DataPoints point : dataSet) {
			labels.add(point.label);
		}
		return labels.size();
	}
	
	public static void writeToFile(Set<DataPoints> noise, List<Set<DataPoints>> clusters, String fileName) {
		// write clusters to file for plotting
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(new File(fileName)));
			for(DataPoints pt : noise) {
				bw.write(pt.x + "," + pt.y + ",0"  + "\n");
			}
			
			for (int w = 0; w < clusters.size(); w++) {
				System.out.println("Cluster " + w + " size :"
						+ clusters.get(w).size());
				for (DataPoints point : clusters.get(w))
					bw.write(point.x + "," + point.y + "," + (w+1) + "\n");
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				bw.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
}