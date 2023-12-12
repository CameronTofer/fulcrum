package main

import (
	_ "embed"
	"fmt"
)



//go:generate git submodule update --init --recursive
//!!git checkout 5f6e0c0dff1e7a89331e6b25eca9a9fd71324069
//go:generate mkdir -p llama.cpp/build
//go:generate cmake -DLLAMA_STATIC=Off -DBUILD_SHARED_LIBS=ON -S llama.cpp -B llama.cpp/build
//go:generate cmake --build llama.cpp/build --config Release
//go:generate cp llama.cpp/build/libllama.dylib .
//go:generate cp llama.cpp/ggml-metal.metal .
//go:generate make -C LuaJIT -j16 BUILDMODE=static MACOSX_DEPLOYMENT_TARGET=14.1
//go:generate make -C LuaJIT install PREFIX=$(PWD)/deps/luajit

/*
#cgo CFLAGS: -Ideps/luajit/include/luajit-2.1
#cgo LDFLAGS: -Ldeps/luajit/lib -lluajit-5.1 -ldl -lm
#include <luajit.h>
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#include "llama.cpp/llama.h"
llama_token llama_sample_logits(float * logits, int n ) {
    float best = -1e20;
    llama_token best_id = 0;
    for (int i=0; i<n; i++) {
        if (logits[i] > best) {
            best = logits[i];
            best_id = i;
        }
    }
    return best_id;
}
*/
import "C"

//go:embed llama.lua
var llamadotlua []byte

func servellm(L *C.lua_State, filename string) func(string) string {

	// load llama.lua
	if 0 != C.luaL_loadbuffer(L, (*C.char)(C.CBytes(llamadotlua)), C.size_t(len(llamadotlua)), C.CString("llama.lua")) {
		panic(C.GoString(C.lua_tolstring(L, -1, nil)))
	}

	if C.lua_pcall(L, 0, -1, 0) != 0 {
		err := C.lua_tolstring(L, -1, nil)
		C.lua_settop(L, 0)
		fmt.Println("error running function `f': ", C.GoString(err))
	}

	// start the llm server
	C.lua_getfield(L, C.LUA_GLOBALSINDEX, C.CString("serve"))
	C.lua_pushstring(L, C.CString(filename))
	if C.lua_resume(L, 1) != C.LUA_YIELD {
		fmt.Println("error on lua_resume", C.GoString(C.lua_tolstring(L, -1, nil)))
		panic("llm server failed to start")
	}

	c := make(chan string)
	go func() {
		for prompt := range c {

			fmt.Println("prompt:", prompt)

			// continue the llm server with the prompt
			C.lua_pushstring(L, C.CString(prompt))
			r := C.lua_resume(L, 1)

			switch r {
			case C.LUA_OK:
				fmt.Println("ok")
			case C.LUA_YIELD:
				answer := C.GoString(C.lua_tolstring(L, -1, nil))
				C.lua_settop(L, 0)
				c <- answer
			default:
				fmt.Println("error on lua_resume:", r, C.GoString(C.lua_tolstring(L, -1, nil)))
			}
		}
	}()
	
	return func(prompt string) string {
		c <- prompt
		return <- c
	}
}

func main() {

	L := C.luaL_newstate()
	defer C.lua_close(L)
	C.luaL_openlibs(L)

	chat := servellm(L, "/Users/cameron/.ollama/models/blobs/sha256:6504ba23a37160de70db611212815c9aab171864d206b8c013b72fd0b16e19eb")


	fmt.Println("How do you cook an egg?\n", chat("How do you cook an egg?"), "===")

	//fmt.Println("What do trains ride on?\n", chat("What do trains ride on?"), "===")
	
}
