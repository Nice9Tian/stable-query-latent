cmake_minimum_required(VERSION 3.16)

project(LariceWindows VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Local toolchain layout mirrors CodeX2Thirdpart: run
# setup_python_pybind11.bat once to populate these two folders.
set(LOCAL_PYTHON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tools/python312")
set(LOCAL_PYBIND11_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11")
set(LARICE_DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dist")
set(LARICE_DIST_RESOURCE_DIR "${LARICE_DIST_DIR}/resources")
set(LARICE_DIST_PYTHON_DIR "${LARICE_DIST_RESOURCE_DIR}/python312")
set(LARICE_DIST_ASSETS_DIR "${LARICE_DIST_RESOURCE_DIR}/assets")

if(EXISTS "${LOCAL_PYTHON_ROOT}/python.exe")
    set(Python_ROOT_DIR "${LOCAL_PYTHON_ROOT}")
    set(Python_EXECUTABLE "${LOCAL_PYTHON_ROOT}/python.exe")
    set(Python_INCLUDE_DIR "${LOCAL_PYTHON_ROOT}/include")
    set(Python_LIBRARY "${LOCAL_PYTHON_ROOT}/libs/python312.lib")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

if(EXISTS "${LOCAL_PYBIND11_ROOT}/CMakeLists.txt")
    add_subdirectory("${LOCAL_PYBIND11_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/external/pybind11")
else()
    find_package(pybind11 REQUIRED)
endif()

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(LariceAnchorStudioApp MANUAL_FINALIZATION ${PROJECT_SOURCES})
else()
    add_executable(LariceAnchorStudioApp ${PROJECT_SOURCES})
endif()

target_link_libraries(LariceAnchorStudioApp PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    pybind11::embed
)

target_compile_definitions(LariceAnchorStudioApp PRIVATE
    LARICE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    LARICE_LOCAL_PYTHON_ROOT="${LOCAL_PYTHON_ROOT}"
)

# Root launcher for the dist package (dist/LariceAnchorStudio.exe ->
# resources/LariceAnchorStudioApp.exe), pattern from CodeX2Thirdpart.
add_executable(LariceAnchorStudio WIN32 launcher.cpp)
target_compile_features(LariceAnchorStudio PRIVATE cxx_std_17)
if(WIN32)
    target_link_libraries(LariceAnchorStudio PRIVATE shell32)
endif()
if(MINGW)
    target_link_options(LariceAnchorStudio PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# Runtime files beside the built exe: python DLLs, the bridge, pip manifest.
if(WIN32 AND EXISTS "${LOCAL_PYTHON_ROOT}/python312.dll")
    add_custom_command(TARGET LariceAnchorStudioApp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/python312.dll"
            "$<TARGET_FILE_DIR:LariceAnchorStudioApp>/python312.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/python3.dll"
            "$<TARGET_FILE_DIR:LariceAnchorStudioApp>/python3.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/larice_bridge.py"
            "$<TARGET_FILE_DIR:LariceAnchorStudioApp>/larice_bridge.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/runtime_requirements.txt"
            "$<TARGET_FILE_DIR:LariceAnchorStudioApp>/runtime_requirements.txt"
    )
endif()

# Deploy the Qt runtime beside the built app for direct Qt Creator release runs.
if(WIN32)
    if(NOT QT_QMAKE_EXECUTABLE)
        get_target_property(QT_QMAKE_EXECUTABLE Qt${QT_VERSION_MAJOR}::qmake IMPORTED_LOCATION)
    endif()
    get_filename_component(QT_BIN_DIR "${QT_QMAKE_EXECUTABLE}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE
        NAMES windeployqt
        HINTS "${QT_BIN_DIR}"
        NO_DEFAULT_PATH
    )
    if(NOT WINDEPLOYQT_EXECUTABLE AND EXISTS "${QT_BIN_DIR}/windeployqt.exe")
        set(WINDEPLOYQT_EXECUTABLE "${QT_BIN_DIR}/windeployqt.exe")
    endif()
    if(NOT WINDEPLOYQT_EXECUTABLE)
        message(FATAL_ERROR "windeployqt was not found in ${QT_BIN_DIR}; cannot create a complete Windows release package.")
    endif()

    add_custom_command(TARGET LariceAnchorStudioApp POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DDEPLOY_CONFIG=$<CONFIG>
            -DWINDEPLOYQT_EXECUTABLE="${WINDEPLOYQT_EXECUTABLE}"
            -DTARGET_FILE="$<TARGET_FILE:LariceAnchorStudioApp>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deploy_qt_runtime.cmake"
        COMMENT "Deploy Qt runtime beside the release app for direct Qt Creator runs"
    )
endif()

# ---------------------------------------------------------------------------
# LariceWindows_release: assemble the fully self-contained dist/ package
#   dist/LariceAnchorStudio.exe          root launcher
#   dist/resources/                      app exe + Qt runtime + python DLLs
#                                        + larice_bridge.py + requirements
#   dist/resources/python312/            complete bundled Python
#   dist/resources/assets/               champion model pack
# ---------------------------------------------------------------------------
if(WIN32 AND EXISTS "${LOCAL_PYTHON_ROOT}/python.exe")
    set(LARICE_ASSET_COMMANDS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/tower.pt")
        list(APPEND LARICE_ASSET_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/tower.pt"
                "${LARICE_DIST_ASSETS_DIR}/tower.pt"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/champion_assets.npz"
                "${LARICE_DIST_ASSETS_DIR}/champion_assets.npz"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/meta.json"
                "${LARICE_DIST_ASSETS_DIR}/meta.json"
        )
    else()
        message(WARNING "windows/assets/tower.pt not found -- the dist package "
                        "will ship WITHOUT the champion model; run "
                        "export_assets.py first, then rebuild LariceWindows_release.")
    endif()

    add_custom_target(LariceWindows_release
        DEPENDS LariceAnchorStudio LariceAnchorStudioApp
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LARICE_DIST_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LARICE_DIST_RESOURCE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LARICE_DIST_PYTHON_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${LARICE_DIST_ASSETS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:LariceAnchorStudio>"
            "${LARICE_DIST_DIR}/$<TARGET_FILE_NAME:LariceAnchorStudio>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:LariceAnchorStudioApp>"
            "${LARICE_DIST_RESOURCE_DIR}/$<TARGET_FILE_NAME:LariceAnchorStudioApp>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/larice_bridge.py"
            "${LARICE_DIST_RESOURCE_DIR}/larice_bridge.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/runtime_requirements.txt"
            "${LARICE_DIST_RESOURCE_DIR}/runtime_requirements.txt"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/python3.dll"
            "${LARICE_DIST_RESOURCE_DIR}/python3.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/python312.dll"
            "${LARICE_DIST_RESOURCE_DIR}/python312.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/vcruntime140.dll"
            "${LARICE_DIST_RESOURCE_DIR}/vcruntime140.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/vcruntime140_1.dll"
            "${LARICE_DIST_RESOURCE_DIR}/vcruntime140_1.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/python.exe"
            "${LARICE_DIST_PYTHON_DIR}/python.exe"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/pythonw.exe"
            "${LARICE_DIST_PYTHON_DIR}/pythonw.exe"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/python3.dll"
            "${LARICE_DIST_PYTHON_DIR}/python3.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/python312.dll"
            "${LARICE_DIST_PYTHON_DIR}/python312.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/vcruntime140.dll"
            "${LARICE_DIST_PYTHON_DIR}/vcruntime140.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_PYTHON_ROOT}/vcruntime140_1.dll"
            "${LARICE_DIST_PYTHON_DIR}/vcruntime140_1.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${LOCAL_PYTHON_ROOT}/DLLs"
            "${LARICE_DIST_PYTHON_DIR}/DLLs"
        COMMAND ${CMAKE_COMMAND}
            -DSRC="${LOCAL_PYTHON_ROOT}/Lib"
            -DDST="${LARICE_DIST_PYTHON_DIR}/Lib"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_python_lib.cmake"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${LOCAL_PYTHON_ROOT}/tcl"
            "${LARICE_DIST_PYTHON_DIR}/tcl"
        ${LARICE_ASSET_COMMANDS}
        COMMAND "${WINDEPLOYQT_EXECUTABLE}"
            --release
            "${LARICE_DIST_RESOURCE_DIR}/$<TARGET_FILE_NAME:LariceAnchorStudioApp>"
        COMMAND_EXPAND_LISTS
        COMMENT "Create dist package with root launcher and bundled app resources"
    )
endif()

set_target_properties(LariceAnchorStudioApp PROPERTIES
    WIN32_EXECUTABLE TRUE
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(LariceAnchorStudioApp)
endif()
