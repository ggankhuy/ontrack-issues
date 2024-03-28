#include <stdio.h>

#include <rdc/rdc.h> //  @manual=//third-party/rocm:librdc
#include <unistd.h>

#include <map>
#include <string>
#include <vector>

std::vector<rdc_field_t> test_fields = {};
std::map<rdc_field_t, std::string> test_fields_names = {};

#define ADD_TEST_FIELD(name)         \
  {                                  \
    test_fields.push_back(name);     \
    test_fields_names[name] = #name; \
  }

int main() {
  ADD_TEST_FIELD(RDC_FI_GPU_COUNT);
  ADD_TEST_FIELD(RDC_FI_DEV_NAME);
  ADD_TEST_FIELD(RDC_FI_GPU_CLOCK);
  ADD_TEST_FIELD(RDC_FI_MEM_CLOCK);
  ADD_TEST_FIELD(RDC_FI_MEMORY_TEMP);
  ADD_TEST_FIELD(RDC_FI_GPU_TEMP);
  ADD_TEST_FIELD(RDC_FI_POWER_USAGE);
  ADD_TEST_FIELD(RDC_FI_PCIE_TX);
  ADD_TEST_FIELD(RDC_FI_PCIE_RX);
  // ADD_TEST_FIELD(RDC_FI_GPU_UTIL);
  // ADD_TEST_FIELD(RDC_FI_GPU_MEMORY_USAGE);
  // ADD_TEST_FIELD(RDC_FI_GPU_MEMORY_TOTAL);
  // ADD_TEST_FIELD(RDC_FI_ECC_CORRECT_TOTAL);
  // ADD_TEST_FIELD(RDC_FI_ECC_UNCORRECT_TOTAL);
  // ADD_TEST_FIELD(RDC_FI_ECC_SDMA_SEC);
  // ADD_TEST_FIELD(RDC_FI_ECC_SDMA_DED);
  // ADD_TEST_FIELD(RDC_FI_ECC_GFX_SEC);
  // ADD_TEST_FIELD(RDC_FI_ECC_GFX_DED);
  // ADD_TEST_FIELD(RDC_FI_ECC_MMHUB_SEC);
  // ADD_TEST_FIELD(RDC_FI_ECC_MMHUB_DED);
  // ADD_TEST_FIELD(RDC_FI_ECC_ATHUB_SEC);
  // ADD_TEST_FIELD(RDC_FI_ECC_ATHUB_DED);
  // ADD_TEST_FIELD(RDC_EVNT_XGMI_0_BEATS_TX);
  // ADD_TEST_FIELD(RDC_EVNT_XGMI_0_THRPUT);
  rdc_handle_t rdc_handle;
  rdc_status_t result = rdc_init(0);
  if (result != RDC_ST_OK) {
    printf("RDC initialization failed with error %d\n", result);
    return -1;
  }
  result = rdc_start_embedded(RDC_OPERATION_MODE_AUTO, &rdc_handle);
  if (result != RDC_ST_OK) {
    printf("RDC start failed with error %d\n", result);
    return -1;
  }
  rdc_gpu_group_t groupId;
  result =
      rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, "MyGroup1", &groupId);
  if (result != RDC_ST_OK) {
    printf("RDC group creation failed with error %d\n", result);
    return -1;
  }
  result = rdc_group_gpu_add(rdc_handle, groupId, 0); // Add GPU 0
  if (result != RDC_ST_OK) {
    printf("failed to add gpu %d to group %d\n", 0, groupId);
    return -1;
  }
  rdc_field_grp_t rdcFieldGroupId;
  result = rdc_group_field_create(
      rdc_handle,
      test_fields.size(),
      test_fields.data(),
      "MyFieldGroup1",
      &rdcFieldGroupId);
  if (result != RDC_ST_OK) {
    printf("RDC field group creation failed with error %d\n", result);
    return -1;
  }
  result = rdc_field_watch(rdc_handle, groupId, rdcFieldGroupId, 1000000, 2, 2);
  if (result != RDC_ST_OK) {
    printf("RDC field watch failed with error %d\n", result);
    return -1;
  }

  while (true) {
    sleep(10);
    rdc_field_value value{};
    for (auto i = 0; i < test_fields.size(); i++) {
      auto field = test_fields[i];
      result = rdc_field_get_latest_value(rdc_handle, 0, field, &value);
      if (result != RDC_ST_OK) {
        printf("RDC field get latest failed with error %d\n", result);
        continue;
      }
      // NOTE rdc_field_get_latest_value() will not set value.status on success.
      if (value.status != RDC_ST_OK) {
        printf("returned value is not RDC_ST_OK\n");
        continue;
      }
      printf("type=%d, updated=%lu\n", value.type, value.ts);
      printf("%s: ", test_fields_names[field].c_str());
      if (value.type == rdc_field_type_t::INTEGER) {
        printf("%ld\n", value.value.l_int);
      } else if (
          value.type == rdc_field_type_t::DOUBLE ||
          value.type == rdc_field_type_t::BLOB) {
        printf("%f\n", value.value.dbl);
      } else if (value.type == rdc_field_type_t::STRING) {
        printf("%s\n", value.value.str);
      }
    }
  }
  return 0;
}
