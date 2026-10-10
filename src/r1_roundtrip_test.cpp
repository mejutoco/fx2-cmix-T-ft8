// Standalone proof for the ported payload_lex transform: reorder -> extract ->
// restore on a copy of the real stream must give back the original byte for
// byte, and must not change the byte vocabulary (the transformer requires 205).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>
#include "r1_reorder_transform.h"
static std::vector<unsigned char> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary); return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)), {});
}
static std::set<unsigned char> vocab(const std::vector<unsigned char>& v) { return std::set<unsigned char>(v.begin(), v.end()); }
int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s <stream-copy-to-mutate> <pristine-stream>\n", argv[0]); return 2; }
  const std::string work = argv[1], side = work + ".side";
  const std::vector<unsigned char> orig = rd(argv[2]);
  const auto v0 = vocab(orig);
  printf("original: %zu B, vocabulary %zu bytes\n", orig.size(), v0.size());
  if (!r1_reorder::ReorderEncodedTailFile(work, side)) { printf("FAIL: reorder\n"); return 1; }
  const std::vector<unsigned char> xf = rd(work);
  const auto v1 = vocab(xf);
  printf("transformed: %zu B (+%zu), vocabulary %zu bytes, side file %zu B\n", xf.size(), xf.size() - orig.size(), v1.size(), rd(side).size());
  if (v1 != v0) { printf("FAIL: vocabulary changed\n"); return 1; }
  if (!r1_reorder::ExtractSideFromFile(work, side + ".x")) { printf("FAIL: extract\n"); return 1; }
  if (!r1_reorder::RestoreEncodedTailFile(work, side + ".x")) { printf("FAIL: restore\n"); return 1; }
  const std::vector<unsigned char> back = rd(work);
  if (back.size() != orig.size() || memcmp(back.data(), orig.data(), orig.size()) != 0) { printf("FAIL: restored stream differs\n"); return 1; }
  printf("OK: reorder -> extract -> restore is exact; vocabulary unchanged (%zu)\n", v0.size());
  return 0;
}
