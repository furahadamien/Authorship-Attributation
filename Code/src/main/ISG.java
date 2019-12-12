package main;

import java.util.*;

public class ISG {
    class ISGEdge {
        String nodeA;
        String nodeB;
        String relation;

        public ISGEdge(String a, String b, String rel) {
            nodeA = a;
            nodeB = b;
            relation = rel;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;

            if (!(obj instanceof ISGEdge))
                return false;

            ISGEdge edge = (ISGEdge) obj;
            if (this.nodeA.equals(edge.nodeA) && this.nodeB.equals(edge.nodeB) && this.relation.equals(edge.relation)) {
                return true;
            } else {
                return false;
            }
        }

        @Override
        public String toString() {
            return relation + "(" + nodeB + ", " + nodeA + ")";
        }
    }

    String root;
    Set<String> nodes;
    Map<String, List<ISGEdge>> edges;

    public ISG() {
        root = "ROOT";
        nodes = new HashSet<>();
        nodes.add(root);
        edges = new HashMap<>();
    }

    public void addEdge(String a, String b, String rel) {
        nodes.add(a); // since it's a set, you can just add the nodes by default and it won't make a duplicate
        nodes.add(b);

        ISGEdge newEdge = new ISGEdge(a, b, rel);
        if (edges.containsKey(a)) {
            edges.get(a).add(newEdge);
        }
        else {
            List<ISGEdge> outgoing = new ArrayList<>();
            outgoing.add(newEdge);
            edges.put(a, outgoing);
        }
    }

    // Looks at the entire ISG and collects all unique feature types, to be returned as a list of strings
    // Uses a LexMorSyn feature set -> unique words, unique POS tags, unique dependencies
    public List<String> getFeatures() {
        List<String> features = new ArrayList<>();

        Set<String> uniqueWords = new HashSet<>();
        Set<String> uniquePOS = new HashSet<>();
        for (String s : nodes) {
            if (s.equals(root))
                continue;
            String[] arr = s.split("/");
            uniqueWords.add(arr[0]);
            uniquePOS.add(arr[1]);
        }

        Set<String> uniqueDep = new HashSet<>();
        for (String n : edges.keySet()) {
            for (ISGEdge e : edges.get(n)) {
                uniqueDep.add(e.relation);
            }
        }

        features.addAll(uniqueWords);
        features.addAll(uniquePOS);
        features.addAll(uniqueDep);
        return features;
    }

    // Computes the feature matrix for an ISG with the following structure:
    //  - rows: the set of nodes (words) other than the ROOT-0 node --> represents the endpoint of the root->node path
    //  - columns: the set of features of interest: word counts, POS tags, dependency tags... TODO add vowel counts
    public HashMap<String, HashMap<String, Integer>> extractFeatureMatrix() {
//        List<String> allFeatures = getFeatures(); // contains three sets: unique words, POS, dep edge labels

        HashMap<String, ArrayList<ISGEdge>> paths = getShortestPaths();
        HashMap<String, HashMap<String, Integer>> matrix = new HashMap<>();
//        for (String n: nodes) {
//            HashMap<String, Integer> hm = new HashMap<>();
//            for (String f : allFeatures) {
//                hm.put(f, 0);
//            }
//
//        }

        for (String n : nodes) {
            if (n.equals(root))
                continue;

            HashMap<String, Integer> vector = new HashMap<>();

            // Loop through the shortest path from ROOT to current node 'n'
            // Tally up counts of each feature type in the path
            ArrayList<ISGEdge> shortestPath = paths.get(n);
            for (ISGEdge step : shortestPath) {
                String[] arr = step.nodeB.split("/");
                String rel = step.relation;

                if (!vector.containsKey(arr[0])) vector.put(arr[0], 0);
                if (!vector.containsKey(arr[1])) vector.put(arr[1], 0);
                if (!vector.containsKey(rel)) vector.put(rel, 0);

                vector.put(arr[0], vector.get(arr[0]) + 1);
                vector.put(arr[1], vector.get(arr[1]) + 1);
                vector.put(rel, vector.get(rel) + 1);
            }

            matrix.put(n, vector);
        }

        return matrix;
    }

    private String getLowestDistance(List<String> nodeList, HashMap<String, Integer> distance) {
        Integer lowest = Integer.MAX_VALUE;
        String minNode = "";

        for (String s : nodeList) {
            if (distance.get(s) < lowest) {
                lowest = distance.get(s);
                minNode = s;
            }
        }

        return minNode;
    }

    public HashMap<String, ArrayList<ISGEdge>> getShortestPaths() {
        HashMap<String, Integer> distance = new HashMap<>();
        for (String n : nodes) {
            distance.put(n, Integer.MAX_VALUE);
        }

        List<String> closed = new ArrayList<>();
        List<String> open = new ArrayList<>();
        HashMap<String, ISGEdge> predecessor = new HashMap<>();

        open.add(root);
        distance.put(root, 0);

        while (open.size() > 0) {
            String curNode = getLowestDistance(open, distance);
            open.remove(curNode);
            closed.add(curNode);

            // Add an empty list of edges to current node if it's a leaf to avoid NPE
            if (edges.get(curNode) == null) {
                edges.put(curNode, new ArrayList<>());
            }

            // Iterate over neighbours of current node
            for (ISGEdge e : edges.get(curNode)) {
                if (!closed.contains(e.nodeB)) {
                    String dest = e.nodeB;
                    int newDist = distance.get(curNode) + 1;
                    if (newDist < distance.get(dest)) {
                        distance.put(dest, newDist);
                        open.add(dest);
                        predecessor.put(dest, e);
                    }
                }
            }
        }

        // Recover the paths
        HashMap<String, ArrayList<ISGEdge>> shortestPaths = new HashMap<>();
        for (String target : nodes) {
            if (target.equals(root))
                continue;

            ArrayList<ISGEdge> path = new ArrayList<>();
            if (predecessor.get(target) == null) {
                shortestPaths.put(target, path);
            } else {
                ISGEdge prev = predecessor.get(target);
                path.add(prev);
                while (predecessor.get(prev.nodeA) != null) {
                    prev = predecessor.get(prev.nodeA);
                    path.add(prev);
                }
                Collections.reverse(path);
                shortestPaths.put(target, path);
            }
        }

        return shortestPaths;
    }

    @Override
    public String toString() {
        String s = nodes.size() + " nodes: [";
        for (String n : nodes)
            s += n + ", ";
        s += "]\n";

        s += " edges: [\n";
        for (String n: edges.keySet()) {
            s += "\t[[" + n + "]] : \t";
            for (ISGEdge outgoing : edges.get(n)) {
                s += outgoing.toString() + ", ";
            }
            s += "\n";
        }
        return s;
    }
}