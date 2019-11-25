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