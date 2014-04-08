
#include <ctype.h>

static const int Control     = 0x1;
static const int Space       = 0x2;
static const int Upper       = 0x4;
static const int Lower       = 0x8;
static const int Punctuation = 0x10;
static const int Digit       = 0x20;
static const int HexDigit    = 0x40;
static const int Break       = 0x80;

static const unsigned char _ascii[] = {
	0,
	Control,	Control,	Control,	Control,	Control,	Control,	Control,	Control,
	Control,	Control|Space,	Control|Space,	Control|Space,	Control|Space,	Control|Space,	Control,	Control,
	Control,	Control,	Control,	Control,	Control,	Control,	Control,	Control,
	Control,	Control,	Control,	Control,	Control,	Control,	Control,	Control,
	Space|Break,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,
	Digit,	Digit,	Digit,	Digit,	Digit,	Digit,	Digit,	Digit,
	Digit,	Digit,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,
	Punctuation,	Upper|HexDigit,	Upper|HexDigit,	Upper|HexDigit,	Upper|HexDigit,	Upper|HexDigit,	Upper|HexDigit,	Upper,
	Upper,	Upper,	Upper,	Upper,	Upper,	Upper,	Upper,	Upper,
	Upper,	Upper,	Upper,	Upper,	Upper,	Upper,	Upper,	Upper,
	Upper,	Upper,	Upper,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,
	Punctuation,	Lower|HexDigit,	Lower|HexDigit,	Lower|HexDigit,	Lower|HexDigit,	Lower|HexDigit,	Lower|HexDigit,	Lower,
	Lower,	Lower,	Lower,	Lower,	Lower,	Lower,	Lower,	Lower,
	Lower,	Lower,	Lower,	Lower,	Lower,	Lower,	Lower,	Lower,
	Lower,	Lower,	Lower,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Control,
	Control,	Control,	Control,	Control,	Control,	Control,	Control,	Control, /* 80 */
	Control,	Control,	Control,	Control,	Control,	Control,	Control,	Control, /* 88 */
	Control,	Control,	Control,	Control,	Control,	Control,	Control,	Control, /* 90 */
	Control,	Control,	Control,	Control,	Control,	Control,	Control,	Control, /* 98 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* A0 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* A8 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* B0 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* B8 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* C0 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* C8 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* D0 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* D8 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* E0 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* E8 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation, /* F0 */
	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation,	Punctuation  /* F8 */
};

extern int isalnum(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Upper | Lower | Digit);
}

extern int isalpha(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Upper | Lower);
}

extern int iscntrl(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Control);
}

extern int isdigit(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Digit);
}

extern int isgraph(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Punctuation | Upper | Lower | Digit);
}

extern int islower(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Lower);
}

extern int isprint(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Punctuation | Upper | Lower | Digit | Break);
}

extern int ispunct(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Punctuation);
}

extern int isspace(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Space);
}

extern int isupper(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Upper);
}

extern int isxdigit(int character)
{
	if(character < 0) return 0;

	return _ascii[(unsigned char)character] & (Digit | HexDigit);
}

extern int isblank(int character)
{
	return character == ' ' || character == '\t';
}

extern int isascii(int character)
{
	return ((unsigned int) character) < 128;
}

extern int tolower(int character)
{
	if(character < 'A') return character;
	if(character > 'Z') return character;

	int difference = 'a' - 'A';
	
	return character + difference;
}

extern int toupper(int character)
{
	if(character < 'a') return character;
	if(character > 'z') return character;
	
	int difference = 'a' - 'A';
	
	return character - difference;
}

