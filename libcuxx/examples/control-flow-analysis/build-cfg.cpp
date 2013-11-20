#include <parallel>
#include <vector>
#include <iostream>
#include <parallel_random>

class Node;

class Edge
{
public:
	Edge(Node* h, Node* t) : head(h), tail(t) {}

public:
	Node* head;
	Node* tail;

};

typedef std::vector<Node>  NodeVector;
typedef std::vector<Edge*> EdgePointerVector;
typedef std::vector<Edge>  EdgeVector;
typedef std::vector<int>   IntVector;


// Lambda?
static void initializeTargetsRandomly(std::parallel_context& context,
	IntVector& targets, int totalNodes)
{
	assert(context.thread_id_in_context() < targets.size());
	
	targets[context.thread_id_in_context()] =
		std::parallel_uniform_random<int>(0, totalNodes);
}

class Node
{
public:
	IntVector targets;
	
public:
	int id;
	EdgePointerVector inEdges;
	EdgePointerVector outEdges;

public:
	void initializeRandomly(int totalNodes)
	{
		int targetCount = std::parallel_uniform_random<int>(0, totalNodes);
		
		targets.resize(targetCount);
		
		std::parallel_launch({targetCount}, initializeTargetsRandomly, targets,
			totalNodes);
	}
	
};

// Lambda?
static void initializeNodesRandomly(std::parallel_context& context,
	NodeVector& nodes)
{
	nodes[context.thread_id_in_context()].initializeRandomly(nodes.size());
}

static void setNodeIds(std::parallel_context& context,
	NodeVector& nodes)
{
	nodes[context.thread_id_in_context()].id = context.thread_id_in_context();
}

class Graph
{
public:
	Graph(int nodeCount)
	: nodes(nodeCount)
	{
		std::parallel_launch({nodes.size()}, setNodeIds, nodes);
	}
	
	void initializeRandomly()
	{
		std::parallel_launch({nodes.size()}, initializeNodesRandomly, nodes);
	}
	
	void controlFlowAnalysis()
	{
		
	}

public:
	NodeVector nodes;
	EdgeVector edges;

};

int main(int argc, char** argv)
{
	Graph graph(1000);
	
	graph.initializeRandomly();
	
	graph.controlFlowAnalysis();
}


