set(patterns
    "*.c"
    "*.h"
    "*.cc"
    "*.hh"
    "*.cpp"
    "*.hpp"
    "*.cxx"
    "*.hxx"
    "*.ipp"
    "*.inl"
    "*.cu"
    "*.cuh"
)

set(found)
foreach(pattern ${patterns})
    file(GLOB_RECURSE matches
        LIST_DIRECTORIES false
        RELATIVE "${PROJECT_ROOT}"
        "${PROJECT_ROOT}/${pattern}"
    )
    list(APPEND found ${matches})
endforeach()

if(found)
    list(SORT found)
    string(REPLACE ";" "\n" found_text "${found}")
    message(FATAL_ERROR "C/C++ extension files remain:\n${found_text}")
endif()
