



int strlen(const char* string)
{
	const char* end = string;

	for( ; *end != '\0'; ++end);
	
	return end - string;
}

bool equal(const char* left, const char* right)
{
	while(true)
	{
		if(*left != *right) return false;

		if(*left == '\0' || *right == '\0') return true;

		++left;
		++right;
	}

	return true;
}

int main(int argc, char** argv)
{
	if(argc > 1 && equal(argv[1], "success"))
		return 1;

	return strlen(argv[0]);
}



