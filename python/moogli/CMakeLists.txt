CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
# setup SIP variables
PROJECT(moogli)

SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_COMPILER "clang++")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_COMPILER "g++")
else()
    message(WARNING "Continuing with unsupported compiler: ${CMAKE_CXX_COMPILER}")
endif()

## Enable/Disable 2011 stupport.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

ADD_DEFINITIONS("-fPIC")

FIND_PACKAGE(PythonInterp REQUIRED)

FIND_PACKAGE(PyQt4 REQUIRED)
SEPARATE_ARGUMENTS(PYQT4_SIP_FLAGS) # convert space separated values to a list

SET(SIP_CONCAT_PARTS 4)
SET(SIP_EXTRA_OPTIONS ${PYQT4_SIP_FLAGS})

IF("${PYQT4_SIP_FLAGS}" STREQUAL "")
    MESSAGE(FATAL_ERROR "++ No PYQT4_SIP_FLAGS found. Can't continue")
ENDIF()
MESSAGE("++ Using sip flags: ${SIP_EXTRA_OPTIONS}")


IF (NOT BINDINGS_GLOBAL_INSTALL)
    SET(PYTHON_SITE_PACKAGES_DIR ${QGIS_DATA_DIR}/python)
ENDIF (NOT BINDINGS_GLOBAL_INSTALL)
IF (WITH_CUSTOM_WIDGETS)
    SET(PYUIC_WIDGET_PLUGIN_DIRECTORY ${PYQT4_MOD_DIR}/uic/widget-plugins/)
ENDIF (WITH_CUSTOM_WIDGETS)

FIND_PACKAGE(Qt4 REQUIRED COMPONENTS QtCore QtGui QtOpenGL)
FIND_PACKAGE(OpenSceneGraph REQUIRED)
FIND_PACKAGE(osgAnimation REQUIRED)

INCLUDE(${QT_USE_FILE})

FIND_PROGRAM(SIP_EXECUTABLE sip)
MESSAGE("++ SIP ${SIP_EXECUTABLE}")

include_directories(./include ${PROJECT_SOURCE_DIR} ".")
include_directories(${OSGQT_INCLUDE_DIR})
include_directories(${SIP_INCLUDES} ${PYTHON_INCLUDE_DIRS})
include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
include_directories(${OSGANIMATION_INCLUDE_DIR})

add_subdirectory(${PROJECT_SOURCE_DIR}/sip)
#add_subdirectory(${PROJECT_SOURCE_DIR}/moc)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

ADD_LIBRARY(moogli SHARED 
    ${PROJECT_SOURCE_DIR}/moc/MorphologyViewer.moc.cpp
    )

SET(MOOGLI_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR}/_moogli)
FILE(MAKE_DIRECTORY ${MOOGLI_BUILD_PATH})

SET_TARGET_PROPERTIES(moogli PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${MOOGLI_BUILD_PATH}"
    PREFIX ""
    )

## Link the following libraries to moogli.
TARGET_LINK_LIBRARIES(moogli
    "-Wl,--whole-archive"
    src
    sip
    "-Wl,--no-whole-archive"
    QtCore
    QtGui
    QtOpenGL           
    osg
    osgFX
    osgUtil
    osgFX
    osgGA
    osgQt
    osgAnimation
    osgViewer
    osgQt
    osgManipulator
    osgText
    )

SET(MOOGLI_TEMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/_moogli")
FILE(MAKE_DIRECTORY ${MOOGLI_TEMP_DIR})

install(FILES ${MOOGLI_BUILD_PATH}/moogli.so
    DESTINATION ${MOOGLI_TEMP_DIR}/moogli
    COMPONENT moogli
    )

install(FILES ${PROJECT_SOURCE_DIR}/__init__.py
    DESTINATION ${MOOGLI_TEMP_DIR}/moogli
    COMPONENT moogli
    )

install(FILES ${PROJECT_SOURCE_DIR}/setup-moogli.py
    DESTINATION ${MOOGLI_TEMP_DIR}
    COMPONENT moogli
    )

install(CODE 
    "execute_process(COMMAND ${PYTHON_EXECUTABLE} setup-moogli.py install
    WORKING_DIRECTORY ${MOOGLI_TEMP_DIR})"
    COMPONENT moogli
    )

SET(CPACK_STRIP_FILES TRUE)
SET(CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE TRUE)
#  We need to compile python scripts on the installation  host.
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Dilawar Singh")
set(CPACK_PACKAGE_CONTACT "dilawars@ncbs.res.in")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "MOOGLI, the visualizer for neural simulator")
set(CPACK_PACKAGE_VERSION "0.9.1")
set(CPACK_SYSTEM_NAME "i386")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "9")
set(CPACK_PACKAGE_VERSION_PATCH "1")
set(CPACK_PACKAGE_VENDOR "NCBS Bangalore")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/README")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Dilawar Singh <dilawars@ncbs.res.in>")
if(WIN32)
    MESSAGE("++ Currently only Unix is supported")
else()
    set(CPACK_STRIP_FILES moose)
endif()
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)

set(CPACK_DEBIAN_PACKAGE_DEPENDS 
    "openscenegraph, python-setuptools, python-qt4, python-sip, libqt4-gui, libopenscenegraph99"
    )

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${PROJECT_SOURCE_DIR}/debian/postinst;${PROJECT_SOURCE_DIR}/debian/prerm;"
    )
INCLUDE(CPack)
