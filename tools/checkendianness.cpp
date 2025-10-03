#include <stdio.h>
//small little endian checker I got from stack overflow (Thank you mr Mehrdad Afshari)
int main(void) {
  int a = 0x12345678;
  unsigned char *c = (unsigned char*) (&a);
  if (*c == 0x78) {
    printf("little-endian\n");
  }
  else {
    printf("big-endian\n");
  }
  return 0;
}
