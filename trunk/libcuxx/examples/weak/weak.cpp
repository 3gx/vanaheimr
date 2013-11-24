
extern __attribute__((weak)) int weakFunction()
{
	return 0;
}

int main(int argc, char** argv)
{
	return weakFunction();
}



