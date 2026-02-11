#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/cereal.hpp>
#include <cereal/external/base64.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "TemplateMatcher.h"
#include "TemplateMatcherC.h"

namespace cereal {
// Saving for cv::Mat
template <class Archive, cereal::traits::DisableIf<
                             cereal::traits::is_text_archive<Archive>::value> =
                             cereal::traits::sfinae>
inline void CEREAL_SAVE_FUNCTION_NAME(Archive& ar, const cv::Mat& mat) {
  int rows, cols, type;
  bool continuous;

  rows = mat.rows;
  cols = mat.cols;
  type = mat.type();
  continuous = mat.isContinuous();

  ar & rows & cols & type & continuous;

  if (continuous) {
    const size_t data_size = mat.total() * mat.elemSize();
    auto mat_data = cereal::binary_data(mat.ptr(), data_size);
    ar & mat_data;
  } else {
    const size_t row_size = cols * mat.elemSize();
    for (int i = 0; i < rows; i++) {
      auto row_data = cereal::binary_data(mat.ptr(i), row_size);
      ar & row_data;
    }
  }
}

// Loading for cv::Mat
template <class Archive, cereal::traits::DisableIf<
                             cereal::traits::is_text_archive<Archive>::value> =
                             cereal::traits::sfinae>
inline void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, cv::Mat& mat) {
  int rows, cols, type;
  bool continuous;

  ar & rows & cols & type & continuous;

  if (continuous) {
    mat.create(rows, cols, type);
    const size_t data_size = mat.total() * mat.elemSize();
    auto mat_data = cereal::binary_data(mat.ptr(), data_size);
    ar & mat_data;
  } else {
    mat.create(rows, cols, type);
    const size_t row_size = cols * mat.elemSize();
    for (int i = 0; i < rows; i++) {
      auto row_data = cereal::binary_data(mat.ptr(i), row_size);
      ar & row_data;
    }
  }
}

template <class Archive, cereal::traits::EnableIf<
                             cereal::traits::is_text_archive<Archive>::value> =
                             cereal::traits::sfinae>
inline void CEREAL_SAVE_FUNCTION_NAME(Archive& ar, const cv::Mat& mat) {
  int rows, cols, type;
  bool continuous;

  rows = mat.rows;
  cols = mat.cols;
  type = mat.type();
  continuous = mat.isContinuous();

  const size_t data_size = mat.total() * mat.elemSize();
  std::vector<unsigned char> buffer;

  if (continuous) {
    buffer.assign(mat.ptr(), mat.ptr() + data_size);
  } else {
    buffer.reserve(data_size);
    const size_t row_size = cols * mat.elemSize();
    for (int i = 0; i < rows; i++) {
      buffer.insert(buffer.end(), mat.ptr(i), mat.ptr(i) + row_size);
    }
  }

  ar(make_nvp("rows", rows), make_nvp("cols", cols), make_nvp("type", type),
     make_nvp("data", cereal::base64::encode(buffer.data(), buffer.size())));
}

template <class Archive, cereal::traits::EnableIf<
                             cereal::traits::is_text_archive<Archive>::value> =
                             cereal::traits::sfinae>
inline void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, cv::Mat& mat) {
  int rows, cols, type;
  std::string encode_data, decode_data;

  ar(make_nvp("rows", rows), make_nvp("cols", cols), make_nvp("type", type),
     make_nvp("data", encode_data));

  decode_data = cereal::base64::decode(encode_data);

  mat.create(rows, cols, type);
  const size_t data_size = mat.total() * mat.elemSize();
  if (decode_data.size() == data_size) {
    std::memcpy(mat.ptr(), decode_data.data(), data_size);
  }
}

// cv::Scalar
template <class Archive>
void CEREAL_SERIALIZE_FUNCTION_NAME(Archive& ar, cv::Scalar& scalar) {
  ar(scalar.val);
}

// TemplData
template <class Archive>
void CEREAL_SERIALIZE_FUNCTION_NAME(Archive& ar,
                                    TemplateMatching::TemplData& t) {
  ar(cereal::make_nvp("vecPyramid", t.vecPyramid),
     cereal::make_nvp("vecTemplMean", t.vecTemplMean),
     cereal::make_nvp("vecTemplNorm", t.vecTemplNorm),
     cereal::make_nvp("vecInvArea", t.vecInvArea),
     cereal::make_nvp("vecResultEqual1", t.vecResultEqual1),
     cereal::make_nvp("bIsPatternLearned", t.bIsPatternLearned),
     cereal::make_nvp("iBorderColor", t.iBorderColor));
}
}  // namespace cereal

namespace TemplateMatching {
class SizeCountingBuffer : public std::streambuf {
 public:
  size_t size = 0;

  SizeCountingBuffer() : size(0) {}

  void reset() { size = 0; }

 protected:
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    size += n;
    return n;
  }

  int_type overflow(int_type c) override {
    if (c != traits_type::eof()) {
      size++;
      return c;
    }
    return traits_type::eof();
  }
};

class MemoryWriteBuffer : public std::streambuf {
 public:
  MemoryWriteBuffer(void* buffer, size_t size) {
    char* start = static_cast<char*>(buffer);
    setp(start, start + size);
  }

  size_t written_size() const { return pptr() - pbase(); }

 protected:
  void safe_pbump(std::streamsize n) {
    while (n > std::numeric_limits<int>::max()) {
      pbump(std::numeric_limits<int>::max());
      n -= std::numeric_limits<int>::max();
    }
    pbump(static_cast<int>(n));
  }

  std::streamsize xsputn(const char* s, std::streamsize n) override {
    std::streamsize avail = epptr() - pptr();
    std::streamsize to_write = (n < avail) ? n : avail;

    if (to_write > 0) {
      std::memcpy(pptr(), s, static_cast<size_t>(to_write));
      safe_pbump(to_write);
    }

    return to_write;
  }

  int_type overflow(int_type c) override {
    if (c != traits_type::eof() && pptr() < epptr()) {
      *pptr() = static_cast<char>(c);
      pbump(1);
      return c;
    }
    return traits_type::eof();
  }

  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which) override {
    if (which & std::ios_base::out) {
      char* target = nullptr;
      if (dir == std::ios_base::beg)
        target = pbase() + off;
      else if (dir == std::ios_base::cur)
        target = pptr() + off;
      else if (dir == std::ios_base::end)
        target = epptr() + off;

      if (target < pbase() || target > epptr()) return pos_type(off_type(-1));
      ptrdiff_t new_pos = target - pbase();
      setp(pbase(), epptr());
      safe_pbump(new_pos);
      return pos_type(new_pos);
    }
    return pos_type(off_type(-1));
  }
};

class MemoryReadBuffer : public std::streambuf {
 public:
  MemoryReadBuffer(const void* buffer, size_t size) {
    char* start = const_cast<char*>(static_cast<const char*>(buffer));
    setg(start, start, start + size);
  }

 protected:
  void safe_gbump(std::streamsize n) {
    while (n > std::numeric_limits<int>::max()) {
      gbump(std::numeric_limits<int>::max());
      n -= std::numeric_limits<int>::max();
    }
    gbump(static_cast<int>(n));
  }

  int_type underflow() override { return traits_type::eof(); }

  std::streamsize xsgetn(char* s, std::streamsize n) override {
    std::streamsize avail = egptr() - gptr();
    std::streamsize to_read = (n < avail) ? n : avail;
    if (to_read > 0) {
      std::memcpy(s, gptr(), static_cast<size_t>(to_read));
      safe_gbump(to_read);
    }
    return to_read;
  }

  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which) override {
    if (which & std::ios_base::in) {
      char* target = nullptr;
      if (dir == std::ios_base::beg)
        target = eback() + off;
      else if (dir == std::ios_base::cur)
        target = gptr() + off;
      else if (dir == std::ios_base::end)
        target = egptr() + off;

      if (target < eback() || target > egptr()) return pos_type(off_type(-1));
      setg(eback(), target, egptr());
      return pos_type(target - eback());
    }
    return pos_type(off_type(-1));
  }

  pos_type seekpos(pos_type sp, std::ios_base::openmode which) override {
    return seekoff(off_type(sp), std::ios_base::beg, which);
  }
};

template <class OutputArchive>
void SaveToStream(std::ostream& os, const TemplData& data) {
  OutputArchive archive(os);
  archive(cereal::make_nvp("TemplData", data));
}

template <class InputArchive>
void LoadFromStream(std::istream& is, TemplData& data) {
  InputArchive archive(is);
  archive(cereal::make_nvp("TemplData", data));
}

bool SaveToStreamByFormat(std::ostream& os, const TemplData& data,
                          TM_Format format) {
  switch (format) {
    case TM_FMT_BINARY:
      SaveToStream<cereal::BinaryOutputArchive>(os, data);
      break;
    case TM_FMT_BINARY_PORTABLE:
      SaveToStream<cereal::PortableBinaryOutputArchive>(os, data);
      break;
    case TM_FMT_JSON:
      SaveToStream<cereal::JSONOutputArchive>(os, data);
      break;
    case TM_FMT_XML:
      SaveToStream<cereal::XMLOutputArchive>(os, data);
      break;
    default:
      return false;
  }
  return true;
}

bool LoadFromStreamByFormat(std::istream& is, TemplData& data,
                            TM_Format format) {
  switch (format) {
    case TM_FMT_BINARY:
      LoadFromStream<cereal::BinaryInputArchive>(is, data);
      break;
    case TM_FMT_BINARY_PORTABLE:
      LoadFromStream<cereal::PortableBinaryInputArchive>(is, data);
      break;
    case TM_FMT_JSON:
      LoadFromStream<cereal::JSONInputArchive>(is, data);
      break;
    case TM_FMT_XML:
      LoadFromStream<cereal::XMLInputArchive>(is, data);
      break;
    default:
      return false;
  }
  return true;
}
}  // namespace TemplateMatching