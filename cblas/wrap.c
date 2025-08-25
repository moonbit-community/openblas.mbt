#include <moonbit.h>
#include <stdlib.h>
#include <string.h>

moonbit_string_t cstr_to_moonbit_string(void *ptr) {
  char *cptr = (char *)ptr;
  int32_t len = strlen(cptr);
  moonbit_string_t ms = moonbit_make_string(len, 0);
  for (int i = 0; i < len; i++) {
    ms[i] = (uint16_t)cptr[i];
  }
  return ms;
}

void free_cstr(char* p) {
  if (p) {
    free(p);
  }
}

void* get_null() {
  return (void*)0;
}

int voidptr_is_null(void* p) {
  return p == NULL;
}
