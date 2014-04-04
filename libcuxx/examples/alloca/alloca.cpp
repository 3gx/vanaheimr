

int main(int argc, char** argv)
{
	char buffer[argc];
	
	for(int i = 0; i < argc; ++i)
	{
		buffer[i] = i;
	}
	
	int sum = 0;

	for(int i = argc - 1; i >= 0; i--)
	{
		sum += buffer[i];
	}
	
	return sum;
}




