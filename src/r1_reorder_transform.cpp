// Payload-lex regime-1 transform for the fixed post-WRT enwik9 tail.
//
// This intentionally implements only the active submission path: reorder the
// known 45,332,670-byte encoded tail, append a compact side permutation, and
// restore it before dictionary decode.

#include "r1_reorder_transform.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace r1_reorder {
namespace {

struct LineRef { size_t start = 0, len = 0; };
struct TailBlock {
  size_t original_index = 0; std::vector<LineRef> lines; std::string sort_key;
};
struct SideMeta {
  size_t r0_count = 0, r1_count = 0, r2_count = 0;
  size_t prelude_count = 0, suffix_count = 0;
  std::vector<size_t> sorted_to_original;
};

const unsigned char kSideMagicD86[] = {'R', '1', 'O', 'R', 'D', '3', '\n'};
const unsigned char kD99Line[] = {0xDF, 0x99, 'N'};
const unsigned char kD86Prefix[] = {0xDF, 0x86, 'N'};

// fx2-cmix-T's stream is 586,459,020 B, 301 B shorter than cmix-lex's; the
// regime-1 boundary content-matches at +0 and the regime-2 boundary at -301,
// and regime 1 holds exactly 243,425 block-shaped D99/D86a lines (verified on
// the real stream before these were set). Regime 0 is copied verbatim, so the
// tail start only has to lie at or before regime 1.
static size_t kEncodedTailStart = 541126651, kEncodedTailLen = 45332369;
static size_t kEncodedRegime1Start = 13599801, kEncodedRegime2Start = 30372587;

// TEST-ONLY overrides, so the runner wiring can be exercised on a small
// synthetic tail instead of a 50 h run. Both FX2T_LEX_TEST=1 and the value
// must be set, and every override is announced on stderr; fx2t-preflight
// refuses to launch a real run with any FX2T_LEX_* variable in the environment.
static void ApplyTestOverrides() {
  const char* t = std::getenv("FX2T_LEX_TEST");
  if (!t || std::strcmp(t, "1") != 0) return;
  auto ov = [](const char* name, size_t* v) {
    const char* e = std::getenv(name);
    if (!e || !*e) return;
    *v = static_cast<size_t>(std::strtoull(e, nullptr, 10));
    std::fprintf(stderr, "*** FX2T_LEX_TEST override %s=%zu ***\n", name, *v);
  };
  ov("FX2T_LEX_TAIL_START", &kEncodedTailStart);
  ov("FX2T_LEX_TAIL_LEN", &kEncodedTailLen);
  ov("FX2T_LEX_R1", &kEncodedRegime1Start);
  ov("FX2T_LEX_R2", &kEncodedRegime2Start);
}

bool SizeTFromU64(uint64_t value, size_t* out) {
  if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;
  *out = static_cast<size_t>(value);
  return true;
}

bool ReadFile(const std::string& path, std::vector<unsigned char>* data) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) return false;
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) return false;
  in.seekg(0, std::ios::beg);
  data->resize(static_cast<size_t>(size));
  if (!data->empty()) in.read(reinterpret_cast<char*>(data->data()), data->size());
  return static_cast<size_t>(in.gcount()) == data->size();
}

bool WriteFile(const std::string& path, const std::vector<unsigned char>& data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) return false;
  if (!data.empty()) out.write(reinterpret_cast<const char*>(data.data()), data.size());
  return out.good();
}

bool HasBytesAt(const std::vector<unsigned char>& data, size_t pos,
    const unsigned char* bytes, size_t len) {
  return pos <= data.size() && data.size() - pos >= len &&
      std::memcmp(data.data() + pos, bytes, len) == 0;
}

size_t BodyEnd(const std::vector<unsigned char>& data, const LineRef& line) {
  if (line.start > data.size() || line.len > data.size() - line.start) return line.start;
  size_t end = line.start + line.len;
  if (end > line.start && data[end - 1] == '\n') --end;
  if (end > line.start && data[end - 1] == '\r') --end;
  return end;
}

std::vector<LineRef> SplitLines(size_t start, size_t len,
    const std::vector<unsigned char>& data) {
  std::vector<LineRef> lines;
  if (start > data.size() || len > data.size() - start) return lines;
  const size_t end = start + len;
  size_t line_start = start;
  for (size_t i = start; i < end; ++i) {
    if (data[i] == '\n') {
      lines.push_back({line_start, i + 1 - line_start});
      line_start = i + 1;
    }
  }
  if (line_start < end) lines.push_back({line_start, end - line_start});
  return lines;
}

bool IsExactD99Line(const std::vector<unsigned char>& data, const LineRef& line) {
  const size_t end = BodyEnd(data, line);
  return end - line.start == sizeof(kD99Line) &&
      std::memcmp(data.data() + line.start, kD99Line, sizeof(kD99Line)) == 0;
}

bool IsAsciiNumberLine(const std::vector<unsigned char>& data, const LineRef& line) {
  const size_t end = BodyEnd(data, line);
  if (line.start == end) return false;
  for (size_t i = line.start; i < end; ++i) {
    const unsigned char c = data[i];
    if ((c < '0' || c > '9') && c != '-') return false;
  }
  return true;
}

bool ParseUnsignedExact(const std::vector<unsigned char>& data, size_t pos,
    size_t end, uint64_t* value) {
  if (pos >= end || data[pos] < '0' || data[pos] > '9') return false;
  uint64_t result = 0;
  while (pos < end && data[pos] >= '0' && data[pos] <= '9') {
    const uint64_t digit = static_cast<uint64_t>(data[pos++] - '0');
    if (result > (std::numeric_limits<uint64_t>::max() - digit) / 10) return false;
    result = result * 10 + digit;
  }
  if (pos != end) return false;
  *value = result;
  return true;
}

bool ParseFirstD86a(const std::vector<unsigned char>& data,
    const TailBlock& block, uint64_t* value) {
  if (block.lines.size() < 2) return false;
  const LineRef& line = block.lines[1];
  const size_t end = BodyEnd(data, line);
  if (end < line.start + sizeof(kD86Prefix) ||
      std::memcmp(data.data() + line.start, kD86Prefix,
          sizeof(kD86Prefix)) != 0) {
    return false;
  }
  return ParseUnsignedExact(data, line.start + sizeof(kD86Prefix), end, value);
}

void AppendLine(std::vector<unsigned char>* out,
    const std::vector<unsigned char>& data, const LineRef& line) {
  out->insert(out->end(), data.begin() + line.start,
      data.begin() + line.start + line.len);
}

void AppendLines(std::vector<unsigned char>* out,
    const std::vector<unsigned char>& data, const std::vector<LineRef>& lines) {
  for (const LineRef& line : lines) AppendLine(out, data, line);
}

void AppendLineToString(std::string* out,
    const std::vector<unsigned char>& data, const LineRef& line) {
  out->append(reinterpret_cast<const char*>(data.data() + line.start),
      line.len);
}

void AppendVarint(std::vector<unsigned char>* out, uint64_t value) {
  while (value >= 0x80) {
    out->push_back(static_cast<unsigned char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out->push_back(static_cast<unsigned char>(value));
}

bool ReadVarint(const std::vector<unsigned char>& data, size_t* pos,
    uint64_t* value) {
  uint64_t result = 0;
  int shift = 0;
  while (*pos < data.size() && shift <= 63) {
    const unsigned char byte = data[(*pos)++];
    if (shift == 63 && (byte & 0x7F) > 1) return false;
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      *value = result;
      return true;
    }
    shift += 7;
  }
  return false;
}


class Fenwick {
 public:
  explicit Fenwick(size_t n) : tree_(n + 1, 0) {
    for (size_t i = 0; i < n; ++i) Add(i, 1);
  }
  void Add(size_t index, int delta) {
    for (++index; index < tree_.size(); index += index & (0 - index)) {
      tree_[index] += delta;
    }
  }
  size_t SumLessThan(size_t index) const {
    size_t result = 0;
    while (index > 0) {
      result += static_cast<size_t>(tree_[index]);
      index -= index & (0 - index);
    }
    return result;
  }
  size_t FindByOrder(size_t rank) const {
    size_t index = 0;
    size_t bit = 1;
    while ((bit << 1) < tree_.size()) bit <<= 1;
    while (bit != 0) {
      const size_t next = index + bit;
      if (next < tree_.size() && static_cast<size_t>(tree_[next]) <= rank) {
        index = next;
        rank -= static_cast<size_t>(tree_[next]);
      }
      bit >>= 1;
    }
    return index;
  }

 private:
  std::vector<int> tree_;
};

bool AppendLehmerPermutation(std::vector<unsigned char>* out,
    const std::vector<size_t>& permutation) {
  Fenwick fenwick(permutation.size());
  std::vector<unsigned char> seen(permutation.size(), 0);
  for (size_t value : permutation) {
    if (value >= permutation.size() || seen[value]) return false;
    seen[value] = 1;
    AppendVarint(out, fenwick.SumLessThan(value));
    fenwick.Add(value, -1);
  }
  return true;
}

bool ReadLehmerPermutation(const std::vector<unsigned char>& data, size_t* pos,
    size_t count, std::vector<size_t>* permutation) {
  Fenwick fenwick(count);
  permutation->assign(count, 0);
  for (size_t i = 0; i < count; ++i) {
    uint64_t rank64 = 0;
    if (!ReadVarint(data, pos, &rank64) || rank64 >= count - i) return false;
    const size_t value = fenwick.FindByOrder(static_cast<size_t>(rank64));
    (*permutation)[i] = value;
    fenwick.Add(value, -1);
  }
  return true;
}

bool PartitionTailLines(const std::vector<LineRef>& lines, size_t* r0_count,
    size_t* r1_count, size_t* r2_count) {
  *r0_count = *r1_count = *r2_count = 0;
  size_t pos = 0;
  for (const LineRef& line : lines) {
    if (pos < kEncodedRegime1Start) ++*r0_count;
    else if (pos < kEncodedRegime2Start) ++*r1_count;
    else ++*r2_count;
    if (line.len > kEncodedTailLen - pos) return false;
    pos += line.len;
  }
  return pos == kEncodedTailLen;
}

size_t GuessLastHugeBlockEnd(const std::vector<unsigned char>& data,
    const std::vector<LineRef>& lines, size_t start, size_t limit) {
  size_t end = std::min(start + static_cast<size_t>(7), limit);
  if (end < limit && IsAsciiNumberLine(data, lines[end])) {
    ++end;
    if (end < limit && !IsExactD99Line(data, lines[end])) {
      const size_t body_end = BodyEnd(data, lines[end]);
      if (body_end > lines[end].start &&
          body_end - lines[end].start <= 16 &&
          data[lines[end].start] >= 0x80) {
        ++end;
      }
    }
  }
  return end;
}

bool ParseTailBlocks(const std::vector<unsigned char>& data,
    const std::vector<LineRef>& r1_lines, std::vector<LineRef>* prelude,
    std::vector<TailBlock>* blocks, std::vector<LineRef>* suffix) {
  std::vector<size_t> starts;
  for (size_t i = 0; i < r1_lines.size(); ++i) {
    if (IsExactD99Line(data, r1_lines[i])) starts.push_back(i);
  }
  if (starts.empty()) return false;
  prelude->assign(r1_lines.begin(), r1_lines.begin() + starts[0]);
  blocks->clear();
  blocks->reserve(starts.size());
  size_t cursor = starts[0];
  for (size_t block_index = 0; block_index < starts.size(); ++block_index) {
    const size_t start = starts[block_index];
    if (start != cursor) return false;
    const size_t next_start =
        block_index + 1 < starts.size() ? starts[block_index + 1]
                                        : r1_lines.size();
    const size_t end = (next_start - start <= 16) ? next_start :
        GuessLastHugeBlockEnd(data, r1_lines, start, next_start);
    TailBlock block;
    block.original_index = block_index;
    block.lines.assign(r1_lines.begin() + start, r1_lines.begin() + end);
    for (size_t i = 2; i < block.lines.size(); ++i) {
      AppendLineToString(&block.sort_key, data, block.lines[i]);
    }
    blocks->push_back(std::move(block));
    cursor = end;
  }
  suffix->assign(r1_lines.begin() + cursor, r1_lines.end());
  return true;
}

bool ComputeD86aOrder(const std::vector<unsigned char>& data,
    const std::vector<TailBlock>& blocks, std::vector<size_t>* order) {
  std::vector<uint64_t> d86a(blocks.size(), 0);
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (!ParseFirstD86a(data, blocks[i], &d86a[i])) return false;
  }
  order->resize(blocks.size());
  std::iota(order->begin(), order->end(), 0);
  std::stable_sort(order->begin(), order->end(), [&d86a](size_t a, size_t b) {
    return d86a[a] != d86a[b] ? d86a[a] < d86a[b] : a < b;
  });
  return true;
}

bool MakeSide(const std::vector<unsigned char>& data,
    const std::vector<TailBlock>& blocks,
    const std::vector<size_t>& sorted_indices, size_t r0_count,
    size_t r1_count, size_t r2_count, size_t prelude_count,
    size_t suffix_count, std::vector<unsigned char>* side) {
  std::vector<uint64_t> d86a(sorted_indices.size(), 0);
  for (size_t sorted_pos = 0; sorted_pos < sorted_indices.size(); ++sorted_pos) {
    const size_t index = sorted_indices[sorted_pos];
    if (index >= blocks.size() ||
        !ParseFirstD86a(data, blocks[index], &d86a[sorted_pos])) {
      return false;
    }
  }
  std::vector<size_t> d86_order;
  d86_order.resize(sorted_indices.size());
  std::iota(d86_order.begin(), d86_order.end(), 0);
  std::stable_sort(d86_order.begin(), d86_order.end(), [&d86a](size_t a, size_t b) {
    return d86a[a] != d86a[b] ? d86a[a] < d86a[b] : a < b;
  });

  std::vector<size_t> original_by_d86a;
  original_by_d86a.reserve(d86_order.size());
  for (size_t sorted_pos : d86_order) {
    original_by_d86a.push_back(sorted_indices[sorted_pos]);
  }

  side->clear();
  side->reserve(32 + original_by_d86a.size() * 3);
  side->insert(side->end(), kSideMagicD86, kSideMagicD86 + sizeof(kSideMagicD86));
  AppendVarint(side, kEncodedTailLen);
  AppendVarint(side, r0_count);
  AppendVarint(side, r1_count);
  AppendVarint(side, r2_count);
  AppendVarint(side, prelude_count);
  AppendVarint(side, suffix_count);
  AppendVarint(side, original_by_d86a.size());
  return AppendLehmerPermutation(side, original_by_d86a);
}

bool ParseSide(const std::vector<unsigned char>& side, SideMeta* meta) {
  if (!HasBytesAt(side, 0, kSideMagicD86, sizeof(kSideMagicD86))) return false;
  size_t pos = sizeof(kSideMagicD86);
  uint64_t tail_len = 0, r0 = 0, r1 = 0, r2 = 0, prelude = 0, suffix = 0;
  uint64_t count = 0;
  if (!ReadVarint(side, &pos, &tail_len) || tail_len != kEncodedTailLen ||
      !ReadVarint(side, &pos, &r0) || !ReadVarint(side, &pos, &r1) ||
      !ReadVarint(side, &pos, &r2) || !ReadVarint(side, &pos, &prelude) ||
      !ReadVarint(side, &pos, &suffix) || !ReadVarint(side, &pos, &count)) {
    return false;
  }
  if (!SizeTFromU64(r0, &meta->r0_count) ||
      !SizeTFromU64(r1, &meta->r1_count) ||
      !SizeTFromU64(r2, &meta->r2_count) ||
      !SizeTFromU64(prelude, &meta->prelude_count) ||
      !SizeTFromU64(suffix, &meta->suffix_count)) {
    return false;
  }
  size_t count_size = 0;
  if (!SizeTFromU64(count, &count_size)) return false;
  if (!ReadLehmerPermutation(side, &pos, count_size,
          &meta->sorted_to_original)) {
    return false;
  }
  return pos == side.size();
}

bool SplitActiveTail(const std::vector<unsigned char>& data,
    std::vector<LineRef>* r0, std::vector<LineRef>* r1,
    std::vector<LineRef>* r2) {
  const std::vector<LineRef> lines =
      SplitLines(kEncodedTailStart, kEncodedTailLen, data);
  size_t r0_count = 0, r1_count = 0, r2_count = 0;
  if (!PartitionTailLines(lines, &r0_count, &r1_count, &r2_count)) return false;
  r0->assign(lines.begin(), lines.begin() + r0_count);
  r1->assign(lines.begin() + r0_count, lines.begin() + r0_count + r1_count);
  r2->assign(lines.begin() + r0_count + r1_count, lines.end());
  return true;
}

}  // namespace

bool ReorderEncodedTailFile(const std::string& path,
    const std::string& side_path) {
  ApplyTestOverrides();
  std::vector<unsigned char> input;
  if (!ReadFile(path, &input)) return false;
  if (input.size() != kEncodedTailStart + kEncodedTailLen) {
    std::fprintf(stderr,
        "\nr1 encoded tail reorder refused: stream=%zu expected=%zu\n",
        input.size(), kEncodedTailStart + kEncodedTailLen);
    return false;
  }

  std::vector<LineRef> r0, r1, r2, prelude, suffix;
  std::vector<TailBlock> blocks;
  if (!SplitActiveTail(input, &r0, &r1, &r2) ||
      !ParseTailBlocks(input, r1, &prelude, &blocks, &suffix)) {
    return false;
  }

  // enwik9 has 243,425 articles and regime 1 holds one block per article.
  // Any other count means the regime constants do not fit this stream;
  // refuse rather than sort the wrong region (the size check above cannot
  // catch an interior shift).
  const char* test_mode = std::getenv("FX2T_LEX_TEST");
  if (!(test_mode && std::strcmp(test_mode, "1") == 0) && blocks.size() != 243425) {
    std::fprintf(stderr, "\nr1 encoded tail reorder refused: parsed %zu blocks, expected 243425\n", blocks.size());
    return false;
  }
  std::vector<size_t> sorted_indices(blocks.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::stable_sort(sorted_indices.begin(), sorted_indices.end(),
      [&blocks](size_t a, size_t b) {
        return blocks[a].sort_key != blocks[b].sort_key ?
            blocks[a].sort_key < blocks[b].sort_key :
            blocks[a].original_index < blocks[b].original_index;
      });

  std::vector<unsigned char> side;
  if (!MakeSide(input, blocks, sorted_indices, r0.size(), r1.size(), r2.size(),
          prelude.size(), suffix.size(), &side) ||
      !WriteFile(side_path, side)) {
    return false;
  }

  std::vector<unsigned char> output;
  output.reserve(input.size() + side.size() * 2 + 32);
  output.insert(output.end(), input.begin(), input.begin() + kEncodedTailStart);
  AppendLines(&output, input, r0);
  AppendLines(&output, input, prelude);
  for (size_t index : sorted_indices) AppendLines(&output, input, blocks[index].lines);
  AppendLines(&output, input, suffix);
  AppendLines(&output, input, r2);
  if (output.size() != input.size()) return false;
  // The side blob is appended inside the stream the transformer reads, and the
  // transformer aborts on any byte outside enwik9's 205-byte vocabulary (cmix-
  // lex's LSTM accepted anything). So the blob travels as lowercase hex plus an
  // ASCII footer line -- bytes that are all in the vocabulary already.
  static const char kHex[] = "0123456789abcdef";
  for (unsigned char c : side) { output.push_back(kHex[c >> 4]); output.push_back(kHex[c & 15]); }
  const std::string footer = "\nR1SJX " + std::to_string(side.size() * 2) + "\n";  // all in the 205-byte vocabulary (E and H are not)
  output.insert(output.end(), footer.begin(), footer.end());
  if (!WriteFile(path, output)) return false;

  std::fprintf(stderr,
      "\nr1 encoded tail reorder: stream=%zu->%zu tail=%zu side=%zu r0=%zu r1=%zu r2=%zu prelude=%zu suffix=%zu blocks=%zu order=payload_lex side=d86a_lehmer stage=post_wrt\n",
      input.size(), output.size(), kEncodedTailLen, side.size(), r0.size(),
      r1.size(), r2.size(), prelude.size(), suffix.size(), blocks.size());
  return true;
}

bool ExtractSideFromFile(const std::string& path,
    const std::string& side_path) {
  std::vector<unsigned char> data;
  if (!ReadFile(path, &data) || data.size() < 16) return false;
  // footer is the last line: "\nR1HEX <hexlen>\n"
  if (data.back() != '\n') return false;
  size_t nl = data.size() - 1;
  while (nl > 0 && data[nl - 1] != '\n') --nl;
  const std::string footer(data.begin() + nl, data.end() - 1);
  if (footer.compare(0, 6, "R1SJX ") != 0) return false;
  const uint64_t hexlen = std::strtoull(footer.c_str() + 6, nullptr, 10);
  if (hexlen == 0 || hexlen % 2 != 0 || hexlen + 1 > nl) return false;
  const size_t hex_pos = nl - 1 - hexlen;  // nl-1 is the '\n' before the footer
  std::vector<unsigned char> side(hexlen / 2);
  auto nib = [](unsigned char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  for (size_t i = 0; i < side.size(); ++i) {
    const int hi = nib(data[hex_pos + 2 * i]), lo = nib(data[hex_pos + 2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    side[i] = static_cast<unsigned char>((hi << 4) | lo);
  }
  data.resize(hex_pos);
  return WriteFile(side_path, side) && WriteFile(path, data);
}

bool RestoreEncodedTailFile(const std::string& path,
    const std::string& side_path) {
  ApplyTestOverrides();
  std::vector<unsigned char> input;
  std::vector<unsigned char> side;
  if (!ReadFile(path, &input) || !ReadFile(side_path, &side)) return false;
  if (input.size() != kEncodedTailStart + kEncodedTailLen) {
    std::fprintf(stderr,
        "\nr1 encoded tail restore refused: stream=%zu expected=%zu\n",
        input.size(), kEncodedTailStart + kEncodedTailLen);
    return false;
  }

  SideMeta meta;
  if (!ParseSide(side, &meta)) return false;
  std::vector<LineRef> lines =
      SplitLines(kEncodedTailStart, kEncodedTailLen, input);
  if (lines.size() != meta.r0_count + meta.r1_count + meta.r2_count) {
    return false;
  }

  std::vector<LineRef> r0(lines.begin(), lines.begin() + meta.r0_count);
  std::vector<LineRef> r1(lines.begin() + meta.r0_count,
      lines.begin() + meta.r0_count + meta.r1_count);
  std::vector<LineRef> r2(lines.begin() + meta.r0_count + meta.r1_count,
      lines.end());
  if (meta.prelude_count + meta.suffix_count > r1.size()) return false;
  std::vector<LineRef> prelude(r1.begin(), r1.begin() + meta.prelude_count);
  std::vector<LineRef> suffix(r1.end() - meta.suffix_count, r1.end());
  std::vector<LineRef> middle(r1.begin() + meta.prelude_count,
      r1.end() - meta.suffix_count);

  std::vector<LineRef> middle_prelude, middle_suffix;
  std::vector<TailBlock> sorted_blocks;
  if (!ParseTailBlocks(input, middle, &middle_prelude, &sorted_blocks,
          &middle_suffix) ||
      !middle_prelude.empty() || !middle_suffix.empty() ||
      sorted_blocks.size() != meta.sorted_to_original.size()) {
    return false;
  }

  std::vector<size_t> d86_order;
  std::vector<const TailBlock*> original_blocks(sorted_blocks.size(), nullptr);
  if (!ComputeD86aOrder(input, sorted_blocks, &d86_order)) return false;
  for (size_t d86_pos = 0; d86_pos < d86_order.size(); ++d86_pos) {
    const size_t sorted_pos = d86_order[d86_pos];
    const size_t original_index = meta.sorted_to_original[d86_pos];
    if (original_index >= original_blocks.size() ||
        original_blocks[original_index]) {
      return false;
    }
    original_blocks[original_index] = &sorted_blocks[sorted_pos];
  }

  std::vector<unsigned char> output;
  output.reserve(input.size());
  output.insert(output.end(), input.begin(), input.begin() + kEncodedTailStart);
  AppendLines(&output, input, r0);
  AppendLines(&output, input, prelude);
  for (const TailBlock* block : original_blocks) {
    if (!block) return false;
    AppendLines(&output, input, block->lines);
  }
  AppendLines(&output, input, suffix);
  AppendLines(&output, input, r2);
  if (output.size() != input.size() || !WriteFile(path, output)) return false;

  std::fprintf(stderr,
      "\nr1 encoded tail restore: stream=%zu tail=%zu side=%zu blocks=%zu order=payload_lex_restored side=%s stage=post_wrt\n",
      output.size(), kEncodedTailLen, side.size(), sorted_blocks.size(),
      "d86a_lehmer");
  return true;
}

}  // namespace r1_reorder
