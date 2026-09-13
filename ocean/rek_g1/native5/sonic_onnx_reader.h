#pragma once

// A bounded reader for the ONNX protobuf fields required by the pinned SONIC
// models. It performs no graph execution and needs no protobuf/Python runtime.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sonic_onnx {

struct Bytes { const unsigned char* data = nullptr; std::size_t size = 0; };
struct Field {
    unsigned number = 0, wire = 0;
    std::uint64_t integer = 0;
    Bytes bytes;
};

class Reader {
    Bytes bytes_;
    std::size_t offset_ = 0;
public:
    explicit Reader(Bytes bytes) : bytes_(bytes) {}
    bool empty() const { return offset_ == bytes_.size; }
    std::uint64_t varint() {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 70; shift += 7) {
            if (offset_ >= bytes_.size) throw std::runtime_error("truncated protobuf varint");
            unsigned byte = bytes_.data[offset_++];
            if (shift == 63 && byte > 1) throw std::runtime_error("protobuf varint overflow");
            value |= std::uint64_t(byte & 127) << shift;
            if ((byte & 128) == 0) return value;
        }
        throw std::runtime_error("protobuf varint overflow");
    }
    Bytes take(std::uint64_t count) {
        if (count > bytes_.size - offset_) throw std::runtime_error("truncated protobuf field");
        Bytes result{bytes_.data + offset_, std::size_t(count)};
        offset_ += std::size_t(count);
        return result;
    }
    bool next(Field& field) {
        if (empty()) return false;
        auto tag = varint();
        if ((tag >> 3) == 0 || (tag >> 3) > 536870911)
            throw std::runtime_error("invalid protobuf field number");
        field = Field{};
        field.number = unsigned(tag >> 3);
        field.wire = unsigned(tag & 7);
        switch (field.wire) {
            case 0: field.integer = varint(); break;
            case 1: field.bytes = take(8); break;
            case 2: field.bytes = take(varint()); break;
            case 5: field.bytes = take(4); break;
            default: throw std::runtime_error("unsupported protobuf wire type");
        }
        return true;
    }
};

inline std::string string_value(const Field& f) {
    if (f.wire != 2) throw std::runtime_error("expected protobuf string");
    return std::string(reinterpret_cast<const char*>(f.bytes.data), f.bytes.size);
}
inline Bytes message_value(const Field& f) {
    if (f.wire != 2) throw std::runtime_error("expected protobuf message");
    return f.bytes;
}

struct Tensor {
    std::string name;
    std::vector<std::int64_t> dimensions;
    std::uint64_t type = 0;
    Bytes raw;
    std::vector<float> float_data;
    bool external = false;
};

inline Tensor tensor(Bytes bytes) {
    Tensor result;
    Reader reader(bytes);
    Field f;
    while (reader.next(f)) {
        if (f.number == 1) {
            if (f.wire == 0) result.dimensions.push_back(std::int64_t(f.integer));
            else if (f.wire == 2) {
                Reader dimensions(f.bytes);
                while (!dimensions.empty()) result.dimensions.push_back(std::int64_t(dimensions.varint()));
            } else throw std::runtime_error("invalid tensor dimensions");
            if (result.dimensions.size() > 16) throw std::runtime_error("tensor rank exceeds limit");
        } else if (f.number == 2 && f.wire == 0) result.type = f.integer;
        else if (f.number == 3) throw std::runtime_error("segmented tensor is unsupported");
        else if (f.number == 4) {
            if ((f.wire != 2 && f.wire != 5) || f.bytes.size % sizeof(float))
                throw std::runtime_error("invalid tensor float_data");
            auto previous = result.float_data.size();
            result.float_data.resize(previous + f.bytes.size / sizeof(float));
            std::memcpy(result.float_data.data() + previous, f.bytes.data, f.bytes.size);
        } else if (f.number == 8) result.name = string_value(f);
        else if (f.number == 9) {
            if (result.raw.data) throw std::runtime_error("duplicate tensor raw_data");
            result.raw = message_value(f);
        } else if (f.number == 13 || (f.number == 14 && f.integer != 0)) result.external = true;
    }
    return result;
}

struct ValueInfo { std::string name; std::uint64_t type = 0; std::vector<std::int64_t> dimensions; };

inline ValueInfo value_info(Bytes bytes) {
    ValueInfo result;
    Reader reader(bytes);
    Field f;
    while (reader.next(f)) {
        if (f.number == 1) result.name = string_value(f);
        if (f.number != 2) continue;
        Reader type(message_value(f));
        Field t;
        while (type.next(t)) {
            if (t.number != 1) continue;
            Reader tensor_type(message_value(t));
            Field tt;
            while (tensor_type.next(tt)) {
                if (tt.number == 1 && tt.wire == 0) result.type = tt.integer;
                if (tt.number != 2) continue;
                Reader shape(message_value(tt));
                Field dimension;
                while (shape.next(dimension)) {
                    if (dimension.number != 1) continue;
                    std::int64_t value = -1;
                    Reader dim(message_value(dimension));
                    Field d;
                    while (dim.next(d)) {
                        if (d.number == 1 && d.wire == 0) value = std::int64_t(d.integer);
                        else if (d.number == 2) throw std::runtime_error("symbolic batch dimension is unsupported");
                    }
                    result.dimensions.push_back(value);
                    if (result.dimensions.size() > 16) throw std::runtime_error("value rank exceeds limit");
                }
            }
        }
    }
    return result;
}

struct Model {
    std::vector<unsigned char> bytes;
    std::map<std::string, Tensor> initializers, constants;
    std::map<std::string, unsigned> operators;
    std::vector<ValueInfo> inputs, outputs;
};

inline void node(Model& model, Bytes bytes) {
    std::string name, operation;
    std::vector<Bytes> attributes;
    Reader reader(bytes);
    Field f;
    while (reader.next(f)) {
        if (f.number == 3) name = string_value(f);
        else if (f.number == 4) operation = string_value(f);
        else if (f.number == 5) attributes.push_back(message_value(f));
        else if (f.number == 7 && !string_value(f).empty())
            throw std::runtime_error("custom ONNX operator domain is unsupported");
    }
    ++model.operators[operation];
    if (operation != "Constant") return;
    bool found = false;
    for (auto bytes : attributes) {
        Reader attribute(bytes);
        Field f;
        while (attribute.next(f)) {
            if (f.number != 5) continue;
            if (found || name.empty()) throw std::runtime_error("invalid ONNX Constant node");
            auto inserted = model.constants.emplace(name, tensor(message_value(f)));
            if (!inserted.second) throw std::runtime_error("duplicate ONNX Constant node");
            found = true;
        }
    }
    if (!found) throw std::runtime_error("Constant node lacks tensor attribute");
}

inline Model load(const char* path) {
    const std::uint32_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_probe) != 1)
        throw std::runtime_error("SONIC model loading requires little endian storage");
    if (!path || !*path) throw std::runtime_error("empty model path");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error(std::string("cannot open model: ") + path);
    auto length = input.tellg();
    if (length <= 0 || length > 256 * 1024 * 1024)
        throw std::runtime_error("model size is outside 1..268435456 bytes");
    Model model;
    model.bytes.resize(std::size_t(length));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(model.bytes.data()), length))
        throw std::runtime_error("model read failed");
    Reader reader({model.bytes.data(), model.bytes.size()});
    Field f;
    bool graph_found = false;
    unsigned opsets = 0;
    while (reader.next(f)) {
        if (f.number == 8) {
            Reader opset(message_value(f));
            Field o;
            std::string domain;
            std::uint64_t version = 0;
            while (opset.next(o)) {
                if (o.number == 1) domain = string_value(o);
                if (o.number == 2 && o.wire == 0) version = o.integer;
            }
            if (!domain.empty() || version != 13) throw std::runtime_error("expected ONNX opset 13");
            ++opsets;
        }
        if (f.number != 7) continue;
        if (graph_found) throw std::runtime_error("duplicate ONNX graph");
        graph_found = true;
        Reader graph(message_value(f));
        Field g;
        while (graph.next(g)) {
            if (g.number == 1) node(model, message_value(g));
            else if (g.number == 5) {
                auto value = tensor(message_value(g));
                auto name = value.name;
                if (name.empty() || !model.initializers.emplace(name, std::move(value)).second)
                    throw std::runtime_error("invalid or duplicate ONNX initializer");
            } else if (g.number == 11) model.inputs.push_back(value_info(message_value(g)));
            else if (g.number == 12) model.outputs.push_back(value_info(message_value(g)));
            else if (g.number == 15) throw std::runtime_error("sparse ONNX initializer is unsupported");
        }
    }
    if (!graph_found || opsets != 1) throw std::runtime_error("invalid ONNX graph/opset count");
    return model;
}

inline void validate_io(const Model& model, std::size_t batch, bool encoder) {
    const auto input_shape = std::vector<std::int64_t>{std::int64_t(batch), encoder ? 1762 : 994};
    const auto output_shape = std::vector<std::int64_t>{std::int64_t(batch), encoder ? 64 : 29};
    if (model.inputs.size() != 1 || model.outputs.size() != 1
        || model.inputs[0].name != "obs_dict" || model.inputs[0].type != 1
        || model.inputs[0].dimensions != input_shape || model.outputs[0].type != 1
        || model.outputs[0].name != (encoder ? "encoded_tokens" : "action")
        || model.outputs[0].dimensions != output_shape)
        throw std::runtime_error("SONIC graph I/O name, type, or exact batch shape mismatch");
    const std::map<std::string, unsigned> encoder_ops = {
        {"Constant",97},{"Reshape",25},{"Mul",19},{"Gemm",15},{"Sigmoid",12},
        {"Slice",11},{"Unsqueeze",8},{"Add",7},{"Concat",7},{"Sub",6},
        {"Shape",5},{"Cast",4},{"ConstantOfShape",3},{"Equal",3},{"Where",3},
        {"Expand",3},{"Tanh",3},{"Round",3},{"Div",3},{"Gather",1},
        {"ScatterND",1},{"Transpose",1},{"ReduceSum",1}};
    const std::map<std::string, unsigned> decoder_ops = {
        {"Constant",15},{"MatMul",7},{"Add",7},{"Sigmoid",6},{"Mul",6},
        {"Slice",3},{"Unsqueeze",2},{"Concat",1},{"Squeeze",1}};
    if (model.operators != (encoder ? encoder_ops : decoder_ops))
        throw std::runtime_error("SONIC operator count contract mismatch");
}

inline std::vector<float> floats(const std::map<std::string, Tensor>& tensors,
    const std::string& name, const std::vector<std::int64_t>& dimensions) {
    auto iterator = tensors.find(name);
    if (iterator == tensors.end()) throw std::runtime_error("missing SONIC tensor: " + name);
    const auto& t = iterator->second;
    if (t.type != 1 || t.external || t.dimensions != dimensions)
        throw std::runtime_error("SONIC tensor shape/type/storage mismatch: " + name);
    std::size_t count = 1;
    for (auto dimension : dimensions) {
        if (dimension <= 0 || std::uint64_t(dimension) > std::numeric_limits<std::size_t>::max() / count)
            throw std::runtime_error("SONIC tensor dimensions overflow");
        count *= std::size_t(dimension);
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::runtime_error("SONIC tensor byte count overflow");
    std::vector<float> result(count);
    if (t.raw.data) {
        if (!t.float_data.empty() || t.raw.size != count * sizeof(float))
            throw std::runtime_error("SONIC tensor raw byte count mismatch: " + name);
        std::memcpy(result.data(), t.raw.data, t.raw.size);
    } else {
        if (t.float_data.size() != count) throw std::runtime_error("SONIC tensor float count mismatch: " + name);
        result = t.float_data;
    }
    for (float value : result) if (!std::isfinite(value))
        throw std::runtime_error("nonfinite SONIC model coefficient: " + name);
    return result;
}

} // namespace sonic_onnx
