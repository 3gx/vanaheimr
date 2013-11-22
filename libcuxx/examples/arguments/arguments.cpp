

int main(int argc, char** argv)
{
	int sum = argc;

	for(int i = 0; i < argc; ++i)
	{
		sum += argv[i][0];
	}

	return sum;
}



