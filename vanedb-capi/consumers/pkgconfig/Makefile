# A consumer that uses only pkg-config. Point PKG_CONFIG_PATH at the extracted
# archive's lib/pkgconfig and run `make test`:
#
#   PKG_CONFIG_PATH=/path/to/vanedb-capi-<version>-<platform>/lib/pkgconfig make -C consumers/pkgconfig test
#
# acceptance_shared links the shared library exactly as `pkg-config --cflags
# --libs vanedb` says, with an rpath so it runs from the archive in place.
# acceptance_static links the static archive by path and takes the system
# libraries from Libs.private (`--static`), which is where the Rust standard
# library's requirements are recorded.
PKG_CONFIG ?= pkg-config
CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror
SOURCE := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../../tests/acceptance.c)
OUT ?= .
LIBDIR := $(shell $(PKG_CONFIG) --variable=libdir vanedb)
STATIC_LIBS := $(filter-out -lvanedb_capi,$(shell $(PKG_CONFIG) --static --libs-only-l --libs-only-other vanedb))

.PHONY: all test clean
all: $(OUT)/acceptance_shared $(OUT)/acceptance_static

$(OUT)/acceptance_shared: $(SOURCE)
	$(CC) $(CFLAGS) $(shell $(PKG_CONFIG) --cflags vanedb) $< -o $@ $(shell $(PKG_CONFIG) --libs vanedb) -Wl,-rpath,$(LIBDIR)

$(OUT)/acceptance_static: $(SOURCE)
	$(CC) $(CFLAGS) $(shell $(PKG_CONFIG) --cflags vanedb) $< -o $@ $(LIBDIR)/libvanedb_capi.a $(STATIC_LIBS)

test: all
	mkdir -p $(OUT)/scratch
	$(OUT)/acceptance_shared $(OUT)/scratch
	$(OUT)/acceptance_static $(OUT)/scratch

clean:
	rm -rf $(OUT)/acceptance_shared $(OUT)/acceptance_static $(OUT)/scratch
