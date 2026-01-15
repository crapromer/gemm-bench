add_rules("mode.debug", "mode.release")

target("matmul_benchmark")
    set_kind("binary")
    set_languages("c++11")
    add_includedirs("include")
    add_files("src/**.cpp")

