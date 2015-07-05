default: release

.PHONY: default release debug all clean

PROJECT_NAME=example

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -Idll/include -Idll/nice_svm/include -Idll/etl/include
LD_FLAGS  += -lsvm -lopencv_core -lopencv_imgproc -lopencv_highgui -ljpeg -lpthread

$(eval $(call use_libcxx))

$(eval $(call auto_folder_compile,src))
$(eval $(call auto_add_executable,$(PROJECT_NAME)))

release: release/bin/$(PROJECT_NAME)
debug: debug/bin/$(PROJECT_NAME)

all: release debug

clean: base_clean

include make-utils/cpp-utils-finalize.mk
