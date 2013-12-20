
#include <cstdio>

class Node
{
public:
	Node(int v, Node* n = nullptr) : value(v), next(n) {}
	~Node() { std::printf(" deleted node %d\n", value); delete next; } 

public:
	void connect(Node* n) { delete next; next = n; }

public:
	int value;
	Node* next;

};

int main(int argc, char** argv)
{
	const int nodes = 50;
	
	Node* root = new Node(0);
	Node* current = root;

	for(unsigned int i = 1; i < nodes; ++i)
	{
		current->connect(new Node(i));
		
		current = current->next;
	}

	current = root;

	std::printf("Visiting list of nodes in order:\n");

	while(current != nullptr)
	{
		std::printf(" visited Node(%d)\n", current->value);

		current = current->next;
	}

	std::printf("Deleting nodes from root:\n");
	delete root;

	return 0;
}



