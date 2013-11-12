#pragma once

typedef struct
{
	int __count;
	union
	{
		wchar_t __wch;
		char __wchb[4];
	} __value;		/* Value so far.  */
} __mbstate_t;

typedef __mbstate_t mbstate_t;


