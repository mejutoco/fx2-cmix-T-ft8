#!/bin/bash -x

set -euo pipefail

#SEED="$1"
#UPDATE_LIMIT="$2"

SEED="923"
UPDATE_LIMIT="3000"

# The decompressor verifies the dictionary it recovers from the archive against
# the size and hash hard-coded in src/readalike_prepr/self_extract.h. Keep the
# two in sync: if the dictionary is ever replaced, that check has to be updated
# or every decompression run will abort.
EXPECTED_DICT_MD5=b3a9cf9fac570b2dc9a0375534792980
actual_dict_md5=$(md5sum dictionary/english.dic | awk '{print $1}')
if [ "$actual_dict_md5" != "$EXPECTED_DICT_MD5" ]; then
  echo "ERROR: dictionary/english.dic has changed (md5 $actual_dict_md5," >&2
  echo "expected $EXPECTED_DICT_MD5). Update kExpectedDictSize and" >&2
  echo "kExpectedDictHash in src/readalike_prepr/self_extract.h (FNV-1a 64) and" >&2
  echo "EXPECTED_DICT_MD5 here." >&2
  exit 1
fi

rm -rf pgo_data 
mkdir -p pgo_data 

# building with PGO
CFLAGS_DEFINES="-DSEED=$SEED -DUPDATE_LIMIT=$UPDATE_LIMIT ${EXTRA_DEFINES:-}"
make CFLAGS_DEFINES="$CFLAGS_DEFINES" prof_gen -j 

./cmix -c ./prof_input/input ./prof_comp > ./prof_output
rm ./prof_comp ./prof_output 
llvm-profdata-17 merge -output=default.profdata ./pgo_data/*
mv default.profdata pgo_data/

make CFLAGS_DEFINES="$CFLAGS_DEFINES" prof_use -j

# Guard against a non-portable binary. The Hutter Prize test machines are an
# Intel Core i7-1165G7 (Tiger Lake: AVX-512, but no AVX-VNNI) and an AMD
# Ryzen 7 (Zen 2: no AVX-512 at all), so anything past the AVX2 baseline is an
# illegal instruction on at least one of them. This catches a stray NATIVE=1,
# a stale objects directory, or a compiler that widens the baseline on its own.
# Checked before upx-ucl, which would hide the code from objdump.
check_avx2_only() {
  local bin="$1" dis bad=0
  if ! command -v objdump > /dev/null; then
    echo "WARNING: objdump not found, skipping the AVX2-only check on $bin" >&2
    return 0
  fi
  dis=$(objdump -d --insn-width=16 "$bin")
  # AVX-512: 512-bit registers, mask registers, or an EVEX (0x62) prefix.
  # AVX-VNNI: objdump prints the VEX encoding of an EVEX-capable opcode as
  # "{vex}", and on Tiger Lake only the EVEX form of vpdpwssd is legal.
  for pat in '%zmm' '%k[0-7]' '\{vex\}' $'\t62 [0-9a-f][0-9a-f] '; do
    if grep -qE "$pat" <<< "$dis"; then
      echo "ERROR: $bin uses instructions beyond the AVX2 baseline (matched '$pat'):" >&2
      grep -E "$pat" <<< "$dis" | head -3 >&2
      bad=1
    fi
  done
  if [ "$bad" -ne 0 ]; then
    echo "Build the submission with the default -march=x86-64-v3 (no NATIVE=1)." >&2
    return 1
  fi
  echo "AVX2-only check passed for $bin"
}
check_avx2_only ./cmix

# Guard against a vendor-dependent binary. RCPPS/RCPSS/RSQRTPS/RSQRTSS are
# reciprocal *estimates*: the standards only bound their relative error
# (1.5*2^-12), the bits they return come from a vendor-specific table, and
# Intel and AMD return different ones. A single differing bit anywhere in the
# model changes a prediction and desynchronises the arithmetic coder, so an
# archive built on an AMD machine decodes to garbage on an Intel one (this is
# what broke the July 2026 submission on the prize committee's Intel laptop).
# The makefile passes -mrecip=none and cpp_infer/src/opt/vec_math.h uses true
# divides; this catches a regression in either.
check_no_recip_estimate() {
  local bin="$1" dis
  if ! command -v objdump > /dev/null; then
    echo "WARNING: objdump not found, skipping the reciprocal-estimate check on $bin" >&2
    return 0
  fi
  dis=$(objdump -d "$bin")
  if grep -qE '\b(v?rcpps|v?rcpss|v?rsqrtps|v?rsqrtss)\b' <<< "$dis"; then
    echo "ERROR: $bin uses reciprocal-estimate instructions, whose results" >&2
    echo "differ between Intel and AMD CPUs:" >&2
    grep -E '\b(v?rcpps|v?rcpss|v?rsqrtps|v?rsqrtss)\b' <<< "$dis" | head -3 >&2
    echo "Build with -mrecip=none and do not use _mm256_rcp_ps/_mm256_rsqrt_ps." >&2
    return 1
  fi
  echo "no reciprocal-estimate instructions in $bin"
}
check_no_recip_estimate ./cmix

 upx-ucl -9 cmix

# this is a directory where the compressor binary will be placed 
DIR=run
mkdir -p ./$DIR
ROOT=$(pwd)
cp ./cmix $DIR/cmix_orig
# git diff > $DIR/patch
# exit
# building a selfextracting binary 
pushd $DIR
# creating a compressed version of dictionary
./cmix_orig -c $ROOT/dictionary/english.dic ./comp_dict
# creating a compressed verions of a file with new order
./cmix_orig -c $ROOT/src/readalike_prepr/data/new_article_order ./comp_order
# the transformer weights are already losslessly compressed (FX2TFWC2, made by
# pysrc/weights_compress.py from the export of models/6m-q4-fp32.tch) and are
# loaded directly in that format, so they are appended as-is
cp $ROOT/models/6m-q4-fp32.tfwc2 ./comp_tfweights
# creating a header with size of the above files
./cmix_orig -h $(wc -c ./comp_dict | awk '{print $1}') $(wc -c ./comp_order | awk '{print $1}') 0 $(wc -c ./comp_tfweights | awk '{print $1}')

# merging the above files and setting permissions for the final executable file
cat ./cmix_orig ./comp_dict ./comp_order ./comp_tfweights header.dat > ./cmix
chmod +x ./cmix
