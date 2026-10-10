// dumps tensor inventory + raw payloads from a compressed weights file
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include "weights_io.h"
int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: %s <compressed> <outdir>\n", argv[0]); return 2; }
  fx2::WeightsFile w = fx2::WeightsFile::load_compressed(argv[1]);
  std::vector<std::string> names;
  for (const auto& kv : w.tensors) names.push_back(kv.first);
  std::sort(names.begin(), names.end());
  const char* dn[4] = {"i8","bf16","f32","i32"};
  size_t tot[4] = {0,0,0,0};
  for (const auto& n : names) {
    const fx2::WTensor& t = w.get(n);
    std::printf("%-6s %10zu %10zu  ", dn[t.dtype], t.numel, t.data.size(), 0);
    for (auto s : t.shape) std::printf("%ux", s);
    std::printf("  %s\n", n.c_str());
    tot[t.dtype] += t.data.size();
    std::string p = std::string(argv[2]) + "/" + n + ".raw";
    for (size_t i = strlen(argv[2]) + 1; i < p.size(); ++i) if (p[i]=='/') p[i]='_';
    FILE* f = fopen(p.c_str(), "wb");
    if (f) { fwrite(t.data.data(), 1, t.data.size(), f); fclose(f); }
  }
  std::printf("\n");
  size_t all = 0;
  for (int i=0;i<4;i++){ std::printf("total %-5s %12zu bytes\n", dn[i], tot[i]); all+=tot[i]; }
  std::printf("total       %12zu bytes, %zu tensors\n", all, names.size());
  return 0;
}
