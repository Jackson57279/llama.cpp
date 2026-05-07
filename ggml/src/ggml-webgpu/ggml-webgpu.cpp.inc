/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#include "ggml-webgpu.h.inc"

#include "ggml-backend-impl-defs.inc"
#include "ggml-impl-defs.inc"
#include "ggml-impl-defs.inc"
#include "ggml-wgsl-shaders.hpp"
#include "ggml.h.inc"
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pre_wgsl {

//==============================================================
// Options
//==============================================================
struct Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
static std::string trim(const std::string & s) {
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char) s[a])) {
        a++;
    }
    size_t b = s.size();
    while (b > a && std::isspace((unsigned char) s[b - 1])) {
        b--;
    }
    return s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::string str;
    std::getline(is, str);
    return trim(str);
}

static bool isIdentChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    if (visiting.count(name)) {
        throw std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    if (it == macros.end()) {
        visiting.erase(name);
        return name;
    }

    const std::string & value = it->second;
    if (value.empty()) {
        visiting.erase(name);
        return "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    return expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    while (i < line.size()) {
        if (isIdentChar(line[i])) {
            size_t start = i;
            while (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            if (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } else {
                result += token;
            }
        } else {
            result += line[i];
            i++;
        }
    }

    return result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    return expandMacrosRecursiveInternal(line, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
class ExprLexer {
  public:
    enum Kind { END, IDENT, NUMBER, OP, LPAREN, RPAREN };

    struct Tok {
        Kind        kind;
        std::string text;
    };

    explicit ExprLexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        if (pos >= src.size()) {
            return { END, "" };
        }

        char c = src[pos];

        // number
        if (std::isdigit((unsigned char) c)) {
            size_t start = pos;
            while (pos < src.size() && std::isdigit((unsigned char) src[pos])) {
                pos++;
            }
            return { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        if (std::isalpha((unsigned char) c) || c == '_') {
            size_t start = pos;
            while (pos < src.size() && (std::isalnum((unsigned char) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            return { IDENT, std::string(src.substr(start, pos - start)) };
        }

        if (c == '(') {
            pos++;
            return { LPAREN, "(" };
        }
        if (c == ')') {
            pos++;
            return { RPAREN, ")" };
        }

        // multi-char operators
        static const char * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        for (auto op : two_ops) {
            if (src.substr(pos, 2) == op) {
                pos += 2;
                return { OP, std::string(op) };
            }
        }

        // single-char operators
        if (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            return { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        return { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        while (pos < src.size() && std::isspace((unsigned char) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
class ExprParser {
  public:
    ExprParser(std::string_view                                     expr,
               const std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(expr),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    int parse() { return parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        if (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            return true;
        }
        return false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        if (tok.kind == k) {
            advance();
            return true;
        }
        return false;
    }

    int parseLogicalOr() {
        int v = parseLogicalAnd();
        while (acceptOp("||")) {
            int rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        return v;
    }

    int parseLogicalAnd() {
        int v = parseEquality();
        while (acceptOp("&&")) {
            int rhs = parseEquality();
            v       = (v && rhs);
        }
        return v;
    }

    int parseEquality() {
        int v = parseRelational();
        for (;;) {
            if (acceptOp("==")) {
                int rhs = parseRelational();
                v       = (v == rhs);
            } else if (acceptOp("!=")) {
                int rhs = parseRelational();
                v       = (v != rhs);
            } else {
                break;
            }
        }
        return v;
    }

    int parseRelational() {
        int v = parseShift();
        for (;;) {
            if (acceptOp("<")) {
                int rhs = parseShift();
                v       = (v < rhs);
            } else if (acceptOp(">")) {
                int rhs = parseShift();
                v       = (v > rhs);
            } else if (acceptOp("<=")) {
                int rhs = parseShift();
                v       = (v <= rhs);
            } else if (acceptOp(">=")) {
                int rhs = parseShift();
                v       = (v >= rhs);
            } else {
                break;
            }
        }
        return v;
    }

    int parseShift() {
        int v = parseAdd();
        for (;;) {
            if (acceptOp("<<")) {
                int rhs = parseAdd();
                v       = (v << rhs);
            } else if (acceptOp(">>")) {
                int rhs = parseAdd();
                v       = (v >> rhs);
            } else {
                break;
            }
        }
        return v;
    }

    int parseAdd() {
        int v = parseMult();
        for (;;) {
            if (acceptOp("+")) {
                int rhs = parseMult();
                v       = (v + rhs);
            } else if (acceptOp("-")) {
                int rhs = parseMult();
                v       = (v - rhs);
            } else {
                break;
            }
        }
        return v;
    }

    int parseMult() {
        int v = parseUnary();
        for (;;) {
            if (acceptOp("*")) {
                int rhs = parseUnary();
                v       = (v * rhs);
            } else if (acceptOp("/")) {
                int rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } else if (acceptOp("%")) {
                int rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } else {
                break;
            }
        }
        return v;
    }

    int parseUnary() {
        if (acceptOp("!")) {
            return !parseUnary();
        }
        if (acceptOp("-")) {
            return -parseUnary();
        }
        if (acceptOp("+")) {
            return +parseUnary();
        }
        return parsePrimary();
    }

    int parsePrimary() {
        // '(' expr ')'
        if (acceptKind(ExprLexer::LPAREN)) {
            int v = parse();
            if (!acceptKind(ExprLexer::RPAREN)) {
                throw std::runtime_error("missing ')'");
            }
            return v;
        }

        // number
        if (tok.kind == ExprLexer::NUMBER) {
            int v = std::stoi(tok.text);
            advance();
            return v;
        }

        // defined(identifier)
        if (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            if (acceptKind(ExprLexer::LPAREN)) {
                if (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                if (!acceptKind(ExprLexer::RPAREN)) {
                    throw std::runtime_error("missing ) in defined()");
                }
                return macros.count(name) ? 1 : 0;
            } else {
                // defined NAME
                if (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                return macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        if (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            if (it == macros.end()) {
                return 0;
            }
            if (it->second.empty()) {
                return 1;
            }
            return evalMacroExpression(name, it->second);
        }

        // unexpected
        return 0;
    }

    int evalMacroExpression(const std::string & name, const std::string & value) {
        if (visiting.count(name)) {
            throw std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        int        v = ep.parse();
        visiting.erase(name);
        return v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        if (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(const std::string & filename, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::All);
        return result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        return result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        return result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        return result;
    }

  private:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    enum class DirectiveMode { All, IncludesOnly };

    struct Cond {
        bool parent_active;
        bool active;
        bool taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    void parseMacroDefinitions(const std::vector<std::string> & macro_defs) {
        for (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            if (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } else {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    void buildMacros(const std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        for (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        for (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            if (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } else {
                name  = trim(def);
                value = "";
            }

            // Add to macros map (will override global if same name)
            macros[name] = value;
            predefined.insert(name);
        }
    }

    //----------------------------------------------------------
    // Helpers
    //----------------------------------------------------------
    std::string loadFile(const std::string & fname) {
        std::ifstream f(fname);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        if (cond.empty()) {
            return true;
        }
        return cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        if (include_stack.count(name)) {
            throw std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        return out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        return processFile(full_path, macros, predefined_macros, include_stack, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(const std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              const std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_stack,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        line;

        while (std::getline(in, line)) {
            std::string t = trim(line);

            if (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                if (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << line << "\n";
                }
            } else {
                if (mode == DirectiveMode::IncludesOnly) {
                    out << line << "\n";
                } else if (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(line, macros);
                    out << expanded << "\n";
                }
            }
        }

        if (mode == DirectiveMode::All && !cond.empty()) {
            throw std::runtime_error("Unclosed #if directive");
        }

        return out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    bool handleDirective(const std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         const std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_stack,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        if (cmd == "include") {
            if (mode == DirectiveMode::All && !condActive(cond)) {
                return true;
            }
            std::string file;
            iss >> file;
            if (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            return true;
        }

        if (mode == DirectiveMode::IncludesOnly) {
            return false;
        }

        if (cmd == "define") {
            if (!condActive(cond)) {
                return true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            if (predefined_macros.count(name)) {
                return true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            return true;
        }

        if (cmd == "undef") {
            if (!condActive(cond)) {
                return true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            if (predefined_macros.count(name)) {
                return true;
            }
            macros.erase(name);
            return true;
        }

        if (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            return true;
        }

        if (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            return true;
        }

        if (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            if (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            return true;
        }

        if (cmd == "elif") {
            std::string expr = trim_value(iss);

            if (cond.empty()) {
                throw std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            if (!c.parent_active) {
                c.active = false;
                return true;
            }

            if (c.taken) {
                c.active = false;
                return true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            if (v) {
                c.taken = true;
            }
            return true;
        }

        if (cmd == "else") {
            if (cond.empty()) {
                throw std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            if (!c.parent_active) {
                c.active = false;
                return true;
            }
            if (c.taken) {
                c.active = false;
            } else {
                c.active = true;
                c.taken  = true;
            }
            return true;
        }

        if (cmd == "endif") {
            if (cond.empty()) {
                throw std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            return true;
        }

        // Unknown directive
        throw std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl



#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define GGML_WEBGPU_F16_SIZE_BYTES                   2
#define GGML_WEBGPU_F32_SIZE_BYTES                   4
#define GGML_WEBGPU_I32_SIZE_BYTES                   4
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#define GGML_WEBGPU_KV_SEQ_PAD                       256u

#define GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE 512u

// Matrix multiplication parameters

// Register tiling parameters
#define WEBGPU_MUL_MAT_TILE_M           4
#define WEBGPU_MUL_MAT_TILE_N           4
#define WEBGPU_MUL_MAT_WG_SIZE_M        8
#define WEBGPU_MUL_MAT_WG_SIZE_N        8
#define WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
#define WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#define WEBGPU_MUL_MAT_SUBGROUP_M            2
// The number of subgroups in the N dimension
#define WEBGPU_MUL_MAT_SUBGROUP_N            4
// The number of subgroup matrices each subgroup accumulates over
#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M     4
#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N     2
#define WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
#define WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32

// Matrix-vector multiplication parameters
#define WEBGPU_MUL_MAT_VEC_WG_SIZE 256

#define WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG    4
#define WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG 4
#define WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG      4

// default size for legacy matrix multiplication
#define WEBGPU_MUL_MAT_WG_SIZE 256

// Same hash combine function as in boost
template <typename T> inline void ggml_webgpu_hash_combine(size_t & seed, const T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Calculates base address of a tensor ignoring the fake base pointer
inline uintptr_t ggml_webgpu_tensor_addr(const ggml_tensor * tensor) {
    const ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    return (uintptr_t) base_tensor->data + tensor->view_offs;
}

inline bool ggml_webgpu_tensor_equal(const ggml_tensor * a, const ggml_tensor * b) {
    return a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
}

inline bool ggml_webgpu_tensor_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    return a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) < ggml_webgpu_tensor_addr(b) + ggml_nbytes(b) &&
           ggml_webgpu_tensor_addr(b) < ggml_webgpu_tensor_addr(a) + ggml_nbytes(a);
}

struct ggml_webgpu_shader_lib_context {
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * src2;
    ggml_tensor * src3;
    ggml_tensor * src4;
    ggml_tensor * src5;
    ggml_tensor * dst;

    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes       = 0;
    bool     supports_subgroups       = false;
    bool     supports_subgroup_matrix = false;
    uint32_t sg_mat_m                 = 0;
    uint32_t sg_mat_n                 = 0;
    uint32_t sg_mat_k                 = 0;
    uint32_t max_subgroup_size        = 0;
};

struct webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<void> context = nullptr;
};

struct ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 0;
    bool     inplace = false;
};

struct ggml_webgpu_binary_shader_decisions {
    uint32_t wg_size     = 0;
    bool     inplace     = false;
    bool     overlap     = false;
    bool     src_overlap = false;
};

struct ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<void> decisions;
};

struct ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_size;
    uint32_t tokens_per_wg;
};

struct ggml_webgpu_ssm_scan_pipeline_key {
    int  type;
    int  d_state;
    bool xbc_overlap;

    bool operator==(const ggml_webgpu_ssm_scan_pipeline_key & other) const {
        return type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap;
    }
};

struct ggml_webgpu_ssm_scan_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_ssm_scan_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.d_state);
        ggml_webgpu_hash_combine(seed, key.xbc_overlap);
        return seed;
    }
};

struct ggml_webgpu_ssm_scan_shader_decisions {
    uint32_t wg_size;
    uint32_t tokens_per_tile;
    bool     xbc_overlap = false;
};

/** Argsort **/

struct ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes;
    int32_t  order;
};

/** Set Rows **/

struct ggml_webgpu_set_rows_pipeline_key {
    int dst_type;
    int vec4;
    int i64_idx;

    bool operator==(const ggml_webgpu_set_rows_pipeline_key & other) const {
        return dst_type == other.dst_type && vec4 == other.vec4 && i64_idx == other.i64_idx;
    }
};

struct ggml_webgpu_set_rows_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_set_rows_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.dst_type);
        ggml_webgpu_hash_combine(seed, key.vec4);
        ggml_webgpu_hash_combine(seed, key.i64_idx);
        return seed;
    }
};

struct ggml_webgpu_set_rows_shader_decisions {
    bool     vec4;
    bool     i64_idx;
    uint32_t wg_size;
};

/** Set **/

struct ggml_webgpu_set_pipeline_key {
    ggml_type type;
    bool      inplace;

    bool operator==(const ggml_webgpu_set_pipeline_key & other) const {
        return type == other.type && inplace == other.inplace;
    }
};

struct ggml_webgpu_set_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_set_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.inplace);
        return seed;
    }
};

/** Get Rows **/

struct ggml_webgpu_get_rows_pipeline_key {
    ggml_type src_type;
    int       vectorized;

    bool operator==(const ggml_webgpu_get_rows_pipeline_key & other) const {
        return src_type == other.src_type && vectorized == other.vectorized;
    }
};

struct ggml_webgpu_get_rows_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_get_rows_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src_type);
        ggml_webgpu_hash_combine(seed, key.vectorized);
        return seed;
    }
};

/** Row Norm **/

struct ggml_webgpu_row_norm_pipeline_key {
    ggml_op op;
    bool    inplace;

    bool operator==(const ggml_webgpu_row_norm_pipeline_key & other) const {
        return op == other.op && inplace == other.inplace;
    }
};

struct ggml_webgpu_row_norm_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_row_norm_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.op);
        ggml_webgpu_hash_combine(seed, key.inplace);
        return seed;
    }
};

/** RMS_NORM + MUL **/

struct ggml_webgpu_rms_norm_mul_pipeline_key {
    bool inplace;      // rn_src == dst
    bool overlap;      // mul_src == dst
    bool src_overlap;  // rn_src == mul_src

    bool operator==(const ggml_webgpu_rms_norm_mul_pipeline_key & other) const {
        return inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

struct ggml_webgpu_rms_norm_mul_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_rms_norm_mul_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.inplace);
        ggml_webgpu_hash_combine(seed, key.overlap);
        ggml_webgpu_hash_combine(seed, key.src_overlap);
        return seed;
    }
};

struct ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size     = 0;
    bool     inplace     = false;
    bool     overlap     = false;
    bool     src_overlap = false;
};

/** Pad **/
struct ggml_webgpu_pad_pipeline_key {
    bool circular;

    bool operator==(const ggml_webgpu_pad_pipeline_key & other) const { return circular == other.circular; }
};

struct ggml_webgpu_pad_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_pad_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.circular);
        return seed;
    }
};

/** Solve Tri **/
struct ggml_webgpu_solve_tri_pipeline_key {
    int type;
    int n;
    int k;

    bool operator==(const ggml_webgpu_solve_tri_pipeline_key & other) const {
        return type == other.type && n == other.n && k == other.k;
    }
};

struct ggml_webgpu_solve_tri_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_solve_tri_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.n);
        ggml_webgpu_hash_combine(seed, key.k);
        return seed;
    }
};

/** SSM Conv **/
struct ggml_webgpu_ssm_conv_pipeline_key {
    int type;
    int vectorized;

    bool operator==(const ggml_webgpu_ssm_conv_pipeline_key & other) const {
        return type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
struct ggml_webgpu_conv2d_pipeline_key {
    ggml_type weight_type;
    ggml_type input_type;
    ggml_type output_type;

    bool operator==(const ggml_webgpu_conv2d_pipeline_key & other) const {
        return weight_type == other.weight_type && input_type == other.input_type && output_type == other.output_type;
    }
};

struct ggml_webgpu_conv2d_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_conv2d_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.weight_type);
        ggml_webgpu_hash_combine(seed, key.input_type);
        ggml_webgpu_hash_combine(seed, key.output_type);
        return seed;
    }
};

/** Im2Col **/
struct ggml_webgpu_im2col_pipeline_key {
    ggml_type input_type;
    ggml_type output_type;

    bool operator==(const ggml_webgpu_im2col_pipeline_key & other) const {
        return input_type == other.input_type && output_type == other.output_type;
    }
};

struct ggml_webgpu_im2col_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_im2col_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.input_type);
        ggml_webgpu_hash_combine(seed, key.output_type);
        return seed;
    }
};

/** Gated Delta Net **/
struct ggml_webgpu_gated_delta_net_pipeline_key {
    int type;
    int s_v;
    int kda;

    bool operator==(const ggml_webgpu_gated_delta_net_pipeline_key & other) const {
        return type == other.type && s_v == other.s_v && kda == other.kda;
    }
};

struct ggml_webgpu_gated_delta_net_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_gated_delta_net_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.s_v);
        ggml_webgpu_hash_combine(seed, key.kda);
        return seed;
    }
};

struct ggml_webgpu_ssm_conv_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_ssm_conv_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.vectorized);
        return seed;
    }
};

/** Scale **/

struct ggml_webgpu_scale_pipeline_key {
    int inplace;

    bool operator==(const ggml_webgpu_scale_pipeline_key & other) const { return inplace == other.inplace; }
};

struct ggml_webgpu_scale_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_scale_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.inplace);
        return seed;
    }
};

/** Upscale **/

struct ggml_webgpu_upscale_pipeline_key {
    ggml_type input_type;
    ggml_type output_type;
    uint32_t  base_mode;
    bool      antialias;

    bool operator==(const ggml_webgpu_upscale_pipeline_key & other) const {
        return input_type == other.input_type && output_type == other.output_type && base_mode == other.base_mode &&
               antialias == other.antialias;
    }
};

struct ggml_webgpu_upscale_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_upscale_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.input_type);
        ggml_webgpu_hash_combine(seed, key.output_type);
        ggml_webgpu_hash_combine(seed, key.base_mode);
        ggml_webgpu_hash_combine(seed, key.antialias);
        return seed;
    }
};

/** Concat **/

struct ggml_webgpu_concat_pipeline_key {
    int type;

    bool operator==(const ggml_webgpu_concat_pipeline_key & other) const { return type == other.type; }
};

struct ggml_webgpu_concat_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_concat_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        return seed;
    }
};

/** Repeat **/

struct ggml_webgpu_repeat_pipeline_key {
    int type;

    bool operator==(const ggml_webgpu_repeat_pipeline_key & other) const { return type == other.type; }
};

struct ggml_webgpu_repeat_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_repeat_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        return seed;
    }
};

/** Binary **/

struct ggml_webgpu_binary_pipeline_key {
    int  type;
    int  op;
    bool inplace;
    bool overlap;
    bool src_overlap;

    bool operator==(const ggml_webgpu_binary_pipeline_key & other) const {
        return type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

struct ggml_webgpu_binary_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_binary_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.op);
        ggml_webgpu_hash_combine(seed, key.inplace);
        ggml_webgpu_hash_combine(seed, key.overlap);
        ggml_webgpu_hash_combine(seed, key.src_overlap);
        return seed;
    }
};

/** Unary **/

struct ggml_webgpu_unary_pipeline_key {
    int           type;
    int           op;
    bool          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    bool          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    bool operator==(const ggml_webgpu_unary_pipeline_key & other) const {
        return type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

struct ggml_webgpu_unary_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_unary_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.op);
        ggml_webgpu_hash_combine(seed, key.is_unary);
        ggml_webgpu_hash_combine(seed, key.inplace);
        ggml_webgpu_hash_combine(seed, key.ttype);
        return seed;
    }
};

/** FlashAttention */

enum ggml_webgpu_flash_attn_path : uint32_t {
    GGML_WEBGPU_FLASH_ATTN_PATH_NONE            = 0u,
    GGML_WEBGPU_FLASH_ATTN_PATH_SUBGROUP_MATRIX = 1u,
    GGML_WEBGPU_FLASH_ATTN_PATH_TILE            = 2u,
    GGML_WEBGPU_FLASH_ATTN_PATH_VEC             = 3u,
};

struct ggml_webgpu_flash_attn_pipeline_key {
    ggml_type kv_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    bool      kv_direct;
    bool      kv_overlap;
    bool      has_mask;
    bool      has_sinks;
    bool      uses_logit_softcap;
    uint32_t  path;

    bool operator==(const ggml_webgpu_flash_attn_pipeline_key & other) const {
        return kv_type == other.kv_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               kv_direct == other.kv_direct && kv_overlap == other.kv_overlap && has_mask == other.has_mask &&
               has_sinks == other.has_sinks && uses_logit_softcap == other.uses_logit_softcap && path == other.path;
    }
};

struct ggml_webgpu_flash_attn_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_flash_attn_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.kv_type);
        ggml_webgpu_hash_combine(seed, key.head_dim_qk);
        ggml_webgpu_hash_combine(seed, key.head_dim_v);
        ggml_webgpu_hash_combine(seed, key.kv_direct);
        ggml_webgpu_hash_combine(seed, key.kv_overlap);
        ggml_webgpu_hash_combine(seed, key.has_mask);
        ggml_webgpu_hash_combine(seed, key.has_sinks);
        ggml_webgpu_hash_combine(seed, key.uses_logit_softcap);
        ggml_webgpu_hash_combine(seed, key.path);
        return seed;
    }
};

struct ggml_webgpu_flash_attn_decisions {
    uint32_t path       = GGML_WEBGPU_FLASH_ATTN_PATH_NONE;
    uint32_t q_tile     = 0;
    uint32_t kv_tile    = 0;
    uint32_t wg_size    = 0;
    bool     kv_direct  = false;
    bool     kv_overlap = false;
};

inline constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
inline constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;

inline uint32_t ggml_webgpu_flash_attn_pick_vec_ne(const ggml_webgpu_flash_attn_pipeline_key & key) {
    if (key.path != GGML_WEBGPU_FLASH_ATTN_PATH_VEC || key.kv_type != GGML_TYPE_F16 ||
        key.head_dim_qk != key.head_dim_v) {
        return 1u;
    }

    switch (key.head_dim_qk) {
        case 64:
        case 192:
        case 576:
            return 2u;
        case 96:
            return 4u;
        default:
            return 1u;
    }
}

inline ggml_webgpu_flash_attn_pipeline_key ggml_webgpu_flash_attn_make_pipeline_key(
    const ggml_webgpu_shader_lib_context & context,
    uint32_t                               path) {
    const bool has_mask  = context.src3 != nullptr;
    const bool has_sinks = context.src4 != nullptr;
    bool       kv_direct = false;
    if (path != GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
        uint32_t kv_direct_align = GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
        if (path == GGML_WEBGPU_FLASH_ATTN_PATH_SUBGROUP_MATRIX) {
            kv_direct_align = context.sg_mat_k;
        }
        kv_direct = (context.src1->type == GGML_TYPE_F16) &&
                    (context.src0->ne[0] % std::max(1u, kv_direct_align) == 0) &&
                    (context.src1->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
    }

    ggml_webgpu_flash_attn_pipeline_key key = {};
    key.kv_type                             = context.src1->type;
    key.head_dim_qk                         = (uint32_t) context.src0->ne[0];
    key.head_dim_v                          = (uint32_t) context.src2->ne[0];
    key.kv_direct                           = kv_direct;
    key.kv_overlap                          = ggml_webgpu_tensor_overlap(context.src1, context.src2);
    key.has_mask                            = has_mask;
    key.has_sinks                           = has_sinks;
    key.uses_logit_softcap                  = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    key.path                                = path;
    return key;
}

struct ggml_webgpu_flash_attn_vec_reduce_pipeline_key {
    uint32_t head_dim_v;
    uint32_t wg_size;
};

struct ggml_webgpu_flash_attn_vec_reduce_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_flash_attn_vec_reduce_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.head_dim_v);
        ggml_webgpu_hash_combine(seed, key.wg_size);
        return seed;
    }
};

inline bool operator==(const ggml_webgpu_flash_attn_vec_reduce_pipeline_key & lhs,
                       const ggml_webgpu_flash_attn_vec_reduce_pipeline_key & rhs) {
    return lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size;
}

struct ggml_webgpu_flash_attn_blk_pipeline_key {
    uint32_t kv_tile;

    bool operator==(const ggml_webgpu_flash_attn_blk_pipeline_key & other) const { return kv_tile == other.kv_tile; }
};

struct ggml_webgpu_flash_attn_blk_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_flash_attn_blk_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.kv_tile);
        return seed;
    }
};

// This is exposed because it's necessary in supports_op
inline size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  bool     has_mask,
                                                  bool     kv_direct) {
    const uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;
    f16_elems += q_tile * head_dim_qk;        // q_shmem
    if (!kv_direct) {
        f16_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f16_elems += q_tile * head_dim_v;         // o_shmem
    if (has_mask) {
        f16_elems += q_tile * kv_tile;        // mask_shmem
    }
    f16_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    return f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

inline uint32_t ggml_webgpu_flash_attn_max_kv_tile(const ggml_webgpu_shader_lib_context &      context,
                                                   const ggml_webgpu_flash_attn_pipeline_key & key) {
    const size_t limit_bytes    = context.wg_mem_limit_bytes;
    uint32_t     q_tile         = context.sg_mat_m;
    uint32_t     kv_granularity = context.sg_mat_n;
    if (key.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
        q_tile         = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
        kv_granularity = std::max(1u, context.max_subgroup_size);
    } else if (key.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
        q_tile         = 1u;
        kv_granularity = 8u;
    }
    const size_t base_q_bytes = (key.head_dim_qk + key.head_dim_v) * q_tile * GGML_WEBGPU_F16_SIZE_BYTES +
                                2 * q_tile * GGML_WEBGPU_F32_SIZE_BYTES;
    size_t       bytes_per_kv = 0;
    if (!key.kv_direct) {
        bytes_per_kv += std::max(key.head_dim_qk, key.head_dim_v);
    }
    if (key.has_mask) {
        bytes_per_kv += q_tile;
    }
    bytes_per_kv += q_tile;
    bytes_per_kv *= GGML_WEBGPU_F16_SIZE_BYTES;
    const uint32_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    return (max_kv_tile / kv_granularity) * kv_granularity;
}

inline ggml_webgpu_flash_attn_decisions ggml_webgpu_flash_attn_get_decisions(
    const ggml_webgpu_shader_lib_context & context,
    size_t                                 storage_offset_alignment) {
    ggml_webgpu_flash_attn_decisions decisions = {};
    const size_t                     alignment = std::max<size_t>(1u, storage_offset_alignment);
    const auto *                     K         = context.src1;
    const auto *                     V         = context.src2;
    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const auto flash_attn_tensor_offset = [](const ggml_tensor * tensor) -> size_t {
        constexpr uintptr_t ptr_base_addr = 0x1000u;
        const ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
        return reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
    };

    const uint32_t k_offset_elems =
        (uint32_t) ((flash_attn_tensor_offset(K) & (alignment - 1)) / ggml_type_size(K->type));
    const uint32_t v_offset_elems =
        (uint32_t) ((flash_attn_tensor_offset(V) & (alignment - 1)) / ggml_type_size(V->type));
    const bool f16_vec4_aligned = (k_offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u) &&
                                  (v_offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u);
    const bool kv_vec_type_supported =
        K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_Q8_0;
    const bool use_vec  = context.supports_subgroups && (context.src0->ne[1] < 20) && (context.src0->ne[0] % 32 == 0) &&
                          (context.src2->ne[0] % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0) &&
                          kv_vec_type_supported && (K->type != GGML_TYPE_F16 || f16_vec4_aligned) &&
                          (context.src2->type == K->type);
    const bool use_tile = context.supports_subgroups && !context.supports_subgroup_matrix && K->type == GGML_TYPE_F16 &&
                          V->type == GGML_TYPE_F16 && f16_vec4_aligned &&
                          (context.src0->ne[0] % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0) &&
                          (context.src2->ne[0] % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0) && !use_vec;

    decisions.path = use_vec                          ? GGML_WEBGPU_FLASH_ATTN_PATH_VEC :
                     use_tile                         ? GGML_WEBGPU_FLASH_ATTN_PATH_TILE :
                     context.supports_subgroup_matrix ? GGML_WEBGPU_FLASH_ATTN_PATH_SUBGROUP_MATRIX :
                                                        GGML_WEBGPU_FLASH_ATTN_PATH_NONE;

    if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_NONE) {
        return decisions;
    }

    const ggml_webgpu_flash_attn_pipeline_key key = ggml_webgpu_flash_attn_make_pipeline_key(context, decisions.path);
    decisions.kv_direct                           = key.kv_direct;
    const uint32_t max_kv_tile                    = ggml_webgpu_flash_attn_max_kv_tile(context, key);
    // invalidate if even the smallest kv_tile doesn't fit in shared memory
    if (max_kv_tile == 0) {
        decisions.path = GGML_WEBGPU_FLASH_ATTN_PATH_NONE;
        return decisions;
    }

    if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
        decisions.q_tile  = 1u;
        decisions.kv_tile = std::max(8u, std::min(32u, max_kv_tile));
        decisions.kv_tile = (decisions.kv_tile / 8u) * 8u;
        decisions.wg_size = std::max(1u, std::min<uint32_t>(32u, context.max_subgroup_size));
        if (decisions.kv_direct) {
            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            while (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
                decisions.kv_tile -= 8u;
            }
        }
        return decisions;
    }

    decisions.q_tile =
        decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE ? GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE : context.sg_mat_m;
    decisions.kv_tile = decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE ?
                            std::min(64u, max_kv_tile) :
                            std::min(max_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES);
    decisions.wg_size = decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE ?
                            GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE :
                            std::max(context.max_subgroup_size, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE);

    if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
        const uint32_t tile_kv_granularity = std::max(1u, context.max_subgroup_size);
        decisions.kv_tile =
            std::max(tile_kv_granularity, (decisions.kv_tile / tile_kv_granularity) * tile_kv_granularity);
    }

    if (decisions.kv_direct) {
        GGML_ASSERT(decisions.kv_tile <= GGML_WEBGPU_KV_SEQ_PAD);
        while (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
            decisions.kv_tile -= decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE ?
                                     std::max(1u, context.max_subgroup_size) :
                                     context.sg_mat_n;
        }
    }
    return decisions;
}

/** Matrix Multiplication **/

struct ggml_webgpu_legacy_mul_mat_pipeline_key {
    ggml_type src0_type;
    ggml_type src1_type;

    bool operator==(const ggml_webgpu_legacy_mul_mat_pipeline_key & other) const {
        return src0_type == other.src0_type && src1_type == other.src1_type;
    }
};

struct ggml_webgpu_legacy_mul_mat_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_legacy_mul_mat_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src0_type);
        ggml_webgpu_hash_combine(seed, key.src1_type);
        return seed;
    }
};

struct ggml_webgpu_mul_mat_vec_pipeline_key {
    ggml_type src0_type;
    ggml_type src1_type;
    int       vectorized;

    bool operator==(const ggml_webgpu_mul_mat_vec_pipeline_key & other) const {
        return src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized;
    }
};

struct ggml_webgpu_mul_mat_vec_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_mul_mat_vec_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src0_type);
        ggml_webgpu_hash_combine(seed, key.src1_type);
        ggml_webgpu_hash_combine(seed, key.vectorized);
        return seed;
    }
};

struct ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
};

struct ggml_webgpu_mul_mat_pipeline_key {
    ggml_type src0_type;
    ggml_type src1_type;
    int       vectorized;
    int       use_subgroup_matrix;

    bool operator==(const ggml_webgpu_mul_mat_pipeline_key & other) const {
        return src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix;
    }
};

struct ggml_webgpu_mul_mat_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_mul_mat_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src0_type);
        ggml_webgpu_hash_combine(seed, key.src1_type);
        ggml_webgpu_hash_combine(seed, key.vectorized);
        ggml_webgpu_hash_combine(seed, key.use_subgroup_matrix);
        return seed;
    }
};

struct ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    int      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_size;
};

/** MUL_MAT_ID **/

struct ggml_webgpu_mul_mat_id_pipeline_key {
    ggml_type src0_type;
    ggml_type src1_type;
    uint32_t  n_experts;
    int       vectorized;

    bool operator==(const ggml_webgpu_mul_mat_id_pipeline_key & other) const {
        return src0_type == other.src0_type && src1_type == other.src1_type && n_experts == other.n_experts &&
               vectorized == other.vectorized;
    }
};

struct ggml_webgpu_mul_mat_id_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_mul_mat_id_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src0_type);
        ggml_webgpu_hash_combine(seed, key.src1_type);
        ggml_webgpu_hash_combine(seed, key.n_experts);
        ggml_webgpu_hash_combine(seed, key.vectorized);
        return seed;
    }
};

/** Cpy **/

struct ggml_webgpu_cpy_pipeline_key {
    ggml_type src_type;
    ggml_type dst_type;

    bool operator==(const ggml_webgpu_cpy_pipeline_key & other) const {
        return src_type == other.src_type && dst_type == other.dst_type;
    }
};

struct ggml_webgpu_cpy_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_cpy_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src_type);
        ggml_webgpu_hash_combine(seed, key.dst_type);
        return seed;
    }
};

/** Glu **/

struct ggml_webgpu_glu_pipeline_key {
    ggml_glu_op glu_op;
    ggml_type   type;
    bool        split;

    bool operator==(const ggml_webgpu_glu_pipeline_key & other) const {
        return glu_op == other.glu_op && type == other.type && split == other.split;
    }
};

struct ggml_webgpu_glu_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_glu_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.glu_op);
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.split);
        return seed;
    }
};

/** Rope **/

struct ggml_webgpu_rope_pipeline_key {
    ggml_type type;
    bool      inplace;
    bool      has_ff;

    bool operator==(const ggml_webgpu_rope_pipeline_key & other) const {
        return type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

struct ggml_webgpu_rope_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_rope_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.inplace);
        ggml_webgpu_hash_combine(seed, key.has_ff);
        return seed;
    }
};

/** SoftMax **/

struct ggml_webgpu_soft_max_pipeline_key {
    ggml_type mask_type;
    bool      has_mask;
    bool      has_sink;
    bool      inplace;

    bool operator==(const ggml_webgpu_soft_max_pipeline_key & other) const {
        return mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

struct ggml_webgpu_soft_max_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_soft_max_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.mask_type);
        ggml_webgpu_hash_combine(seed, key.has_mask);
        ggml_webgpu_hash_combine(seed, key.has_sink);
        ggml_webgpu_hash_combine(seed, key.inplace);
        return seed;
    }
};

class ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<int, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<int, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<int, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<int, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<int, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_key, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_key_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_key, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_key_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_key, webgpu_pipeline, ggml_webgpu_unary_pipeline_key_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_key, webgpu_pipeline, ggml_webgpu_scale_pipeline_key_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_key, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_key_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_key, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_key_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_key, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_key_hash>
        ssm_scan_pipelines;   // type/d_state
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_key_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_key, webgpu_pipeline, ggml_webgpu_pad_pipeline_key_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_binary_pipeline_key, webgpu_pipeline, ggml_webgpu_binary_pipeline_key_hash>
        binary_pipelines;           // type/op/inplace/overlap
    std::unordered_map<ggml_webgpu_concat_pipeline_key, webgpu_pipeline, ggml_webgpu_concat_pipeline_key_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_key, webgpu_pipeline, ggml_webgpu_repeat_pipeline_key_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_key, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_key_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_key_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_key_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_legacy_mul_mat_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_legacy_mul_mat_pipeline_key_hash>
        mul_mat_legacy_pipelines;  // legacy mul_mat (non-subgroup/non-regtile/non-vec)
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_key_hash>
        mul_mat_vec_pipelines;     // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_key, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_key_hash>
                                             mul_mat_fast_pipelines;       // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<int, webgpu_pipeline> mul_mat_id_gather_pipelines;  // key is fixed
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_key, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_key_hash>
        mul_mat_id_pipelines;                                              // src0_type/src1_type
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_key, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_key_hash>
        mul_mat_id_vec_pipelines;                                          // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_key, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_key_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_key, webgpu_pipeline, ggml_webgpu_set_pipeline_key_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_key, webgpu_pipeline, ggml_webgpu_cpy_pipeline_key_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_key, webgpu_pipeline, ggml_webgpu_glu_pipeline_key_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_key, webgpu_pipeline, ggml_webgpu_rope_pipeline_key_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_key, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_key_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_key, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_key_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_key, webgpu_pipeline, ggml_webgpu_im2col_pipeline_key_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_key_hash>
        rms_norm_mul_pipelines;
    std::unordered_map<ggml_webgpu_upscale_pipeline_key, webgpu_pipeline, ggml_webgpu_upscale_pipeline_key_hash>
        upscale_pipelines;

  public:
    ggml_webgpu_shader_lib(wgpu::Device device) { this->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(const ggml_webgpu_shader_lib_context & context) {
        auto it = sum_rows_pipelines.find(1);
        if (it != sum_rows_pipelines.end()) {
            return it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        return sum_rows_pipelines[1];
    }

    webgpu_pipeline get_row_norm_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_key key = {};
        key.op                                = context.dst->op;
        key.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = row_norm_pipelines.find(key);
        if (it != row_norm_pipelines.end()) {
            return it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        switch (key.op) {
            case GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                break;
            case GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                break;
            default:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        const uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size              = wg_size;
        decisions->inplace              = key.inplace;
        row_norm_pipelines[key]         = ggml_webgpu_create_pipeline(device, processed, variant);
        row_norm_pipelines[key].context = decisions;
        return row_norm_pipelines[key];
    }

    webgpu_pipeline get_argmax_pipeline(const ggml_webgpu_shader_lib_context & context) {
        bool vec4 = context.src0->ne[0] % 4 == 0;

        auto it = argmax_pipelines.find(vec4);
        if (it != argmax_pipelines.end()) {
            return it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        if (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        auto processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        return argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_rows_pipeline_key key = {};
        key.dst_type                          = context.dst->type;
        key.vec4                              = context.src0->ne[0] % 4 == 0;
        key.i64_idx                           = context.src1->type == GGML_TYPE_I64;

        auto it = set_rows_pipelines.find(key);
        if (it != set_rows_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        switch (context.dst->type) {
            case GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_dstf32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_dstf16";
                break;
            default:
                GGML_ABORT("Unsupported dst type for set_rows shader");
        }

        if (key.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        if (key.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_set_rows, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = key.vec4;
        decisions->i64_idx              = key.i64_idx;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[key]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[key].context = decisions;
        return set_rows_pipelines[key];
    }

    webgpu_pipeline get_set_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_key key = {};
        key.type                         = context.dst->type;
        key.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = set_pipelines.find(key);
        if (it != set_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        switch (key.type) {
            case GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                break;
            default:
                GGML_ABORT("Unsupported type for set shader");
        }

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_set, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = key.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[key]       = pipeline;
        return set_pipelines[key];
    }

    webgpu_pipeline get_cumsum_pipeline(const ggml_webgpu_shader_lib_context & context) {
        auto it = cumsum_pipelines.find(1);
        if (it != cumsum_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        return cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(const ggml_webgpu_shader_lib_context & context) {
        bool          is_top_k = context.dst->op == GGML_OP_TOP_K;
        // ascending order is 0, descending order is 1
        const int32_t order =
            is_top_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_pipelines.find(order);
        if (it != argsort_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        while (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        auto decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        return argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(const ggml_webgpu_shader_lib_context & context) {
        bool          is_top_k = context.dst->op == GGML_OP_TOP_K;
        // ascending order is 0, descending order is 1
        const int32_t order =
            is_top_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_merge_pipelines.find(order);
        if (it != argsort_merge_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        return argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(const ggml_webgpu_shader_lib_context & context) {
        const bool vectorized                 = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0;
        ggml_webgpu_get_rows_pipeline_key key = {};
        key.src_type                          = context.src0->type;
        key.vectorized                        = (int) vectorized;

        auto it = get_rows_pipelines.find(key);
        if (it != get_rows_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        const struct ggml_type_traits * type_traits = ggml_get_type_traits(key.src_type);
        const char *                    type_str    = type_traits->type_name;

        switch (key.src_type) {
            case GGML_TYPE_F32:
                defines.push_back("FLOAT_PARALLEL");
                if (key.vectorized) {
                    defines.push_back("F32_VEC");
                    defines.push_back("SRC_TYPE=vec4<f32>");
                    defines.push_back("DST_TYPE=vec4<f32>");
                    defines.push_back("BLOCK_SIZE=4u");
                } else {
                    defines.push_back("F32");
                    defines.push_back("SRC_TYPE=f32");
                    defines.push_back("DST_TYPE=f32");
                    defines.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                break;
            case GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                break;
            default:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    switch (key.src_type) {
                        case GGML_TYPE_Q1_0:
                        case GGML_TYPE_Q4_0:
                        case GGML_TYPE_Q5_0:
                        case GGML_TYPE_Q8_0:
                        case GGML_TYPE_Q3_K:
                        case GGML_TYPE_Q6_K:
                        case GGML_TYPE_IQ2_XXS:
                        case GGML_TYPE_IQ2_XS:
                        case GGML_TYPE_IQ2_S:
                        case GGML_TYPE_IQ3_XXS:
                        case GGML_TYPE_IQ3_S:
                        case GGML_TYPE_IQ1_S:
                        case GGML_TYPE_IQ4_NL:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                break;
                            }
                        default:
                            {
                                defines.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back(type_upper + "_T");
                    defines.push_back(type_upper);
                    defines.push_back(type_upper + "_SCALE_MIN");
                    defines.push_back(type_upper + "_TABLES");
                    defines.push_back(type_upper + "_GRID");

                    variant += "_";
                    variant += type_str;

                    defines.push_back("DST_TYPE=f32");

                    if (key.src_type == GGML_TYPE_Q1_0) {
                        defines.push_back("BLOCK_SIZE=128u");
                    } else if ((key.src_type >= GGML_TYPE_Q4_0 && key.src_type <= GGML_TYPE_Q8_1) ||
                               key.src_type == GGML_TYPE_IQ4_NL) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } else if (key.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } else {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    break;
                }
        }

        if (key.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[key]  = pipeline;
        return get_rows_pipelines[key];
    }

    webgpu_pipeline get_scale_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_key key = {};
        key.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = scale_pipelines.find(key);
        if (it != scale_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_scale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = key.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[key]     = pipeline;
        return scale_pipelines[key];
    }

    webgpu_pipeline get_solve_tri_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_key key = {};
        key.type                               = context.dst->type;
        key.n                                  = (int) context.src0->ne[0];
        key.k                                  = (int) context.src1->ne[0];

        auto it = solve_tri_pipelines.find(key);
        if (it != solve_tri_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        switch (key.type) {
            case GGML_TYPE_F32:
                variant += "_f32";
                break;
            default:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        const uint32_t wg_size       = std::min((uint32_t) key.n, context.max_wg_size);
        const uint32_t k_tile        = wg_size;
        const uint32_t bytes_per_row = ((uint32_t) key.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        const uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(key.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        auto processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[key] = pipeline;
        return solve_tri_pipelines[key];
    }

    webgpu_pipeline get_ssm_conv_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_key key = {};
        key.type                              = context.dst->type;
        key.vectorized                        = context.src1->ne[0] == 4;

        auto it = ssm_conv_pipelines.find(key);
        if (it != ssm_conv_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        switch (key.type) {
            case GGML_TYPE_F32:
                variant += "_f32";
                break;
            default:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        if (key.vectorized) {
            defines.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t block_size    = 32u;
        constexpr uint32_t tokens_per_wg = 8u;

        defines.push_back("BLOCK_SIZE=" + std::to_string(block_size) + "u");
        defines.push_back("TOKENS_PER_WG=" + std::to_string(tokens_per_wg) + "u");

        auto processed           = preprocessor.preprocess(wgsl_ssm_conv, defines);
        auto decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->block_size    = block_size;
        decisions->tokens_per_wg = tokens_per_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[key]  = pipeline;
        return ssm_conv_pipelines[key];
    }

    webgpu_pipeline get_ssm_scan_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_scan_pipeline_key key = {};
        key.type                              = context.dst->type;
        key.d_state                           = (int) context.src0->ne[0];
        key.xbc_overlap                       = ggml_webgpu_tensor_overlap(context.src1, context.src4) &&
                                                ggml_webgpu_tensor_overlap(context.src1, context.src5);

        auto it = ssm_scan_pipelines.find(key);
        if (it != ssm_scan_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_scan";

        switch (key.type) {
            case GGML_TYPE_F32:
                variant += "_f32";
                break;
            default:
                GGML_ABORT("Unsupported type for ssm_scan shader");
        }

        const uint32_t wg_size = (uint32_t) key.d_state;

        constexpr uint32_t tokens_per_tile = 4u;

        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");

        if (context.supports_subgroups) {
            defines.push_back("USE_SUBGROUP_REDUCTION");
            variant += "_sg_reduce";
        } else {
            variant += "_wg_reduce";
        }

        if (key.xbc_overlap) {
            defines.push_back("XBC_OVERLAP");
        }

        variant += "_d" + std::to_string(key.d_state);

        auto processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
        auto decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
        decisions->wg_size         = wg_size;
        decisions->tokens_per_tile = tokens_per_tile;
        decisions->xbc_overlap     = key.xbc_overlap;
        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        ssm_scan_pipelines[key]    = pipeline;
        return ssm_scan_pipelines[key];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_key key = {};
        key.type                                     = context.dst->type;
        key.s_v                                      = (int) context.src2->ne[0];
        key.kda                                      = context.src3->ne[0] == context.src2->ne[0];

        auto it = gated_delta_net_pipelines.find(key);
        if (it != gated_delta_net_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        switch (key.type) {
            case GGML_TYPE_F32:
                variant += "_f32";
                break;
            default:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        if (key.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        defines.push_back("S_V=" + std::to_string(key.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(key.s_v) + "u");

        auto            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[key] = pipeline;
        return gated_delta_net_pipelines[key];
    }

    webgpu_pipeline get_pad_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_key key = {};
        key.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

        auto it = pad_pipelines.find(key);
        if (it != pad_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        if (key.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_pad, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[key]       = pipeline;
        return pad_pipelines[key];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_vec_pipeline_key key = {};
        key.src0_type                            = context.src0->type;
        key.src1_type                            = context.src1->type;
        key.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_vec_pipelines.find(key);
        if (it != mul_mat_vec_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        const char *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        switch (context.src0->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                variant += "_f16";
                break;
            default:
                {
                    // Quantized types: use helpers but accumulate in f16
                    const struct ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    switch (context.src0->type) {
                        case GGML_TYPE_IQ1_S:
                        case GGML_TYPE_IQ1_M:
                        case GGML_TYPE_IQ2_S:
                        case GGML_TYPE_IQ3_S:
                        case GGML_TYPE_IQ4_NL:
                        case GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            break;
                        case GGML_TYPE_IQ2_XXS:
                        case GGML_TYPE_IQ2_XS:
                        case GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            break;
                        default:
                            break;
                    }
                    break;
                }
        }

        // src1 type (vector)
        switch (context.src1->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                break;
            default:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // VEC/SCALAR controls
        defines.push_back(key.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        if (key.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } else if (key.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } else if (key.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        if (key.vectorized) {
            variant += "_vectorized";
        }

        auto processed            = preprocessor.preprocess(shader_src, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = key.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[key] = pipeline;
        return mul_mat_vec_pipelines[key];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_pipeline_key key = {};
        key.src0_type           = context.src0->type;
        key.src1_type           = context.src1->type;
        key.vectorized          = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                                   (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                      1 :
                                      0;
        key.use_subgroup_matrix = context.supports_subgroup_matrix;

        auto it = mul_mat_fast_pipelines.find(key);
        if (it != mul_mat_fast_pipelines.end()) {
            return it->second;
        }

        const char * shader_src = key.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = key.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        switch (context.src1->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                break;
            default:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        const struct ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        const char *                    src0_name   = src0_traits->type_name;

        switch (context.src0->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("FLOAT");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("FLOAT");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                break;
            default:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    switch (context.src0->type) {
                        case GGML_TYPE_IQ1_S:
                        case GGML_TYPE_IQ1_M:
                        case GGML_TYPE_IQ4_NL:
                        case GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            break;
                        case GGML_TYPE_IQ2_XXS:
                        case GGML_TYPE_IQ2_XS:
                        case GGML_TYPE_IQ2_S:
                        case GGML_TYPE_IQ3_XXS:
                        case GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            break;
                        default:
                            break;
                    }

                    variant += std::string("_") + src0_name;
                    break;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(key.vectorized ? "VEC" : "SCALAR");

        const bool is_quant = ggml_is_quantized(context.src0->type);

        uint32_t tile_k;
        if (key.use_subgroup_matrix) {
            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
        } else {
            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
        }

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");

        // Subgroup matrix specifics
        if (key.use_subgroup_matrix) {
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            defines.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            defines.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            defines.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M) + "u");
            defines.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N) + "u");
            defines.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            defines.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            defines.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        if (key.vectorized) {
            variant += "_vectorized";
        }

        if (!key.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
        }

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = tile_k;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = key.use_subgroup_matrix;
        if (key.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_size           = context.max_subgroup_size;
        } else {
            decisions->wg_size_m       = WEBGPU_MUL_MAT_WG_SIZE_M;
            decisions->wg_size_n       = WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->wg_size         = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->mul_mat_wg_size = WEBGPU_MUL_MAT_WG_SIZE;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[key] = pipeline;
        return mul_mat_fast_pipelines[key];
    }

    webgpu_pipeline get_mul_mat_legacy_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_legacy_mul_mat_pipeline_key key = {};
        key.src0_type                               = context.src0->type;
        key.src1_type                               = context.src1->type;

        auto it = mul_mat_legacy_pipelines.find(key);
        if (it != mul_mat_legacy_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat";

        switch (context.src1->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC1_TYPE=f32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC1_TYPE=f16");
                variant += "_f16";
                break;
            default:
                GGML_ABORT("Unsupported src1 type for mul_mat legacy shader");
        }

        const struct ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        const char *                    src0_name   = src0_traits->type_name;

        switch (context.src0->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC0_TYPE=f32");
                defines.push_back("FLOAT");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC0_TYPE=f16");
                defines.push_back("FLOAT");
                variant += "_f16";
                break;
            default:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    switch (context.src0->type) {
                        case GGML_TYPE_Q4_0:
                        case GGML_TYPE_Q5_0:
                        case GGML_TYPE_Q8_0:
                        case GGML_TYPE_Q3_K:
                        case GGML_TYPE_Q6_K:
                        case GGML_TYPE_IQ2_XXS:
                        case GGML_TYPE_IQ2_XS:
                        case GGML_TYPE_IQ2_S:
                        case GGML_TYPE_IQ3_XXS:
                        case GGML_TYPE_IQ3_S:
                        case GGML_TYPE_IQ1_S:
                        case GGML_TYPE_IQ4_NL:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC0_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                break;
                            }
                        default:
                            {
                                defines.push_back(std::string("SRC0_TYPE=") + src0_name);
                            }
                    }

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back(type_upper + "_T");
                    defines.push_back(type_upper);
                    defines.push_back(type_upper + "_SCALE_MIN");
                    defines.push_back(type_upper + "_TABLES");
                    defines.push_back(type_upper + "_GRID");

                    variant += std::string("_") + src0_name;
                    break;
                }
        }

        auto processed = preprocessor.preprocess(wgsl_mul_mat, defines);

        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = WEBGPU_MUL_MAT_WG_SIZE;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_legacy_pipelines[key] = pipeline;
        return mul_mat_legacy_pipelines[key];
    }

    webgpu_pipeline get_mul_mat_id_gather_pipeline(const ggml_webgpu_shader_lib_context & context) {
        auto it = mul_mat_id_gather_pipelines.find(1);
        if (it != mul_mat_id_gather_pipelines.end()) {
            return it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context               = decisions;
        mul_mat_id_gather_pipelines[1] = pipeline;
        return pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_key key = {};
        key.src0_type                           = context.src0->type;
        key.src1_type                           = context.src1->type;
        key.n_experts                           = context.src0->ne[2];
        key.vectorized = (context.src0->ne[0] % 4 == 0 && context.src0->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_pipelines.find(key);
        if (it != mul_mat_id_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

        // src1 type
        switch (context.src1->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                break;
            default:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        const struct ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        const char *                    src0_name   = src0_traits->type_name;

        switch (context.src0->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                break;
            default:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    switch (context.src0->type) {
                        case GGML_TYPE_IQ1_S:
                        case GGML_TYPE_IQ1_M:
                        case GGML_TYPE_IQ4_NL:
                        case GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            break;
                        case GGML_TYPE_IQ2_XXS:
                        case GGML_TYPE_IQ2_XS:
                        case GGML_TYPE_IQ2_S:
                        case GGML_TYPE_IQ3_XXS:
                        case GGML_TYPE_IQ3_S:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            break;
                        default:
                            break;
                    }

                    variant += std::string("_") + src0_name;
                    break;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(key.vectorized ? "VEC" : "SCALAR");

        // mul_mat_id is register-tile only.
        const uint32_t tile_k =
            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        if (key.vectorized) {
            variant += "_vectorized";
        }

        auto processed = preprocessor.preprocess(wgsl_mul_mat_id, defines);

        auto decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = tile_k;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_size_m = WEBGPU_MUL_MAT_WG_SIZE_M;
        decisions->wg_size_n = WEBGPU_MUL_MAT_WG_SIZE_N;
        decisions->wg_size   = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[key] = pipeline;
        return mul_mat_id_pipelines[key];
    }

    webgpu_pipeline get_mul_mat_id_vec_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_key key = {};
        key.src0_type                           = context.src0->type;
        key.src1_type                           = context.src1->type;
        key.n_experts                           = context.src0->ne[2];
        key.vectorized = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;

        auto it = mul_mat_id_vec_pipelines.find(key);
        if (it != mul_mat_id_vec_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_id_vec";
        const char *             shader_src = wgsl_mul_mat_id_vec;

        // src1 type
        switch (context.src1->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                break;
            default:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        switch (context.src0->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                variant += "_f16";
                break;
            default:
                {
                    // Quantized types: use helpers but accumulate in f16
                    const struct ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    switch (context.src0->type) {
                        case GGML_TYPE_IQ1_S:
                        case GGML_TYPE_IQ1_M:
                        case GGML_TYPE_IQ2_S:
                        case GGML_TYPE_IQ3_S:
                        case GGML_TYPE_IQ4_NL:
                        case GGML_TYPE_IQ4_XS:
                            defines.push_back(type_upper + "_GRID");
                            break;
                        case GGML_TYPE_IQ2_XXS:
                        case GGML_TYPE_IQ2_XS:
                        case GGML_TYPE_IQ3_XXS:
                            defines.push_back(type_upper + "_GRID");
                            defines.push_back(type_upper + "_TABLES");
                            break;
                        default:
                            break;
                    }
                    break;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(key.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        if (key.src0_type == GGML_TYPE_Q1_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        } else if (key.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } else if (key.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        if (key.vectorized) {
            variant += "_vectorized";
        }

        defines.push_back(std::string("N_EXPERTS=") + std::to_string(key.n_experts));

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_id_vec_pipelines[key] = pipeline;
        return mul_mat_id_vec_pipelines[key];
    }

    webgpu_pipeline get_unary_pipeline(const ggml_webgpu_shader_lib_context & context) {
        const bool                     is_unary = context.dst->op == GGML_OP_UNARY;
        const int                      op       = is_unary ? (int) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_key key      = {};
        key.type                                = context.dst->type;
        key.op                                  = op;
        key.is_unary                            = is_unary;
        key.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
        key.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

        auto it = unary_pipelines.find(key);
        if (it != unary_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            key.is_unary ? ggml_unary_op_name((ggml_unary_op) key.op) : ggml_op_name((ggml_op) key.op);
        defines.push_back(variant);

        switch (key.type) {
            case GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                break;
            default:
                GGML_ABORT("Unsupported type for unary shader");
        }

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        if (op == GGML_OP_TRI) {
            switch (key.ttype) {
                case GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    break;
                case GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    break;
                case GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    break;
                case GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    break;
                default:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_unary, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = key.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[key]     = pipeline;
        return unary_pipelines[key];
    }

    webgpu_pipeline get_rms_norm_mul_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rms_norm_mul_pipeline_key key = {};
        key.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
        key.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
        key.src_overlap                           = ggml_webgpu_tensor_overlap(context.src0, context.src1);

        auto it = rms_norm_mul_pipelines.find(key);
        if (it != rms_norm_mul_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } else if (key.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } else if (key.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = key.inplace;
        pipeline_decisions->overlap     = key.overlap;
        pipeline_decisions->src_overlap = key.src_overlap;
        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context                = pipeline_decisions;
        rms_norm_mul_pipelines[key]     = pipeline;
        return rms_norm_mul_pipelines[key];
    }

    webgpu_pipeline get_binary_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_binary_pipeline_key key = {};
        key.type                            = context.dst->type;
        key.op                              = context.dst->op;
        key.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
        key.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
        key.src_overlap                     = ggml_webgpu_tensor_overlap(context.src0, context.src1);

        auto it = binary_pipelines.find(key);
        if (it != binary_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) key.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        switch (key.type) {
            case GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                break;
            default:
                GGML_ABORT("Unsupported type for binary shader");
        }

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } else if (key.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } else if (key.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_binary, defines);
        auto pipeline_decisions         = std::make_shared<ggml_webgpu_binary_shader_decisions>();
        pipeline_decisions->wg_size     = context.max_wg_size;
        pipeline_decisions->inplace     = key.inplace;
        pipeline_decisions->overlap     = key.overlap;
        pipeline_decisions->src_overlap = key.src_overlap;

        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = pipeline_decisions;
        binary_pipelines[key]    = pipeline;
        return binary_pipelines[key];
    }

    webgpu_pipeline get_concat_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_concat_pipeline_key key = {};
        key.type                            = context.dst->type;

        auto it = concat_pipelines.find(key);
        if (it != concat_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        switch (key.type) {
            case GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                break;
            default:
                GGML_ABORT("Unsupported type for concat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_concat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[key]    = pipeline;
        return concat_pipelines[key];
    }

    webgpu_pipeline get_repeat_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_key key = {};
        key.type                            = context.dst->type;

        auto it = repeat_pipelines.find(key);
        if (it != repeat_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        switch (key.type) {
            case GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                break;
            case GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                break;
            default:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_repeat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[key]    = pipeline;
        return repeat_pipelines[key];
    }

    webgpu_pipeline get_flash_attn_pipeline(const ggml_webgpu_shader_lib_context & context,
                                            size_t                                 storage_offset_alignment) {
        const ggml_webgpu_flash_attn_decisions decisions =
            ggml_webgpu_flash_attn_get_decisions(context, storage_offset_alignment);
        GGML_ASSERT(decisions.path != GGML_WEBGPU_FLASH_ATTN_PATH_NONE);
        ggml_webgpu_flash_attn_pipeline_key key = ggml_webgpu_flash_attn_make_pipeline_key(context, decisions.path);
        auto                                it  = flash_attn_pipelines.find(key);
        if (it != flash_attn_pipelines.end()) {
            return it->second;
        }
        std::vector<std::string> defines;
        std::string              variant = decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC  ? "flash_attn_vec" :
                                           decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE ? "flash_attn_tile" :
                                                                                                "flash_attn";

        switch (key.kv_type) {
            case GGML_TYPE_F32:
                defines.push_back("KV_F32");
                break;
            case GGML_TYPE_F16:
                defines.push_back("KV_F16");
                break;
            case GGML_TYPE_Q4_0:
                defines.push_back("KV_Q4_0");
                break;
            case GGML_TYPE_Q8_0:
                defines.push_back("KV_Q8_0");
                break;
            default:
                GGML_ABORT("Unsupported KV type for flash attention shader");
        }
        variant += std::string("_") + ggml_type_name(key.kv_type);

        if (key.has_mask) {
            defines.push_back("MASK");
            if (key.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
                defines.push_back("BLK");
                variant += "_mask_blk";
            } else {
                variant += "_mask";
            }
        }
        if (key.has_sinks) {
            defines.push_back("SINKS");
            variant += "_sinks";
        }
        if (key.uses_logit_softcap) {
            defines.push_back("LOGIT_SOFTCAP");
            variant += "_lgsc";
        }
        if (key.kv_direct) {
            defines.push_back("KV_DIRECT");
            variant += "_kvdirect";
        }
        if (key.kv_overlap) {
            defines.push_back("KV_OVERLAP");
            variant += "_kv_overlap";
        }

        defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(key.head_dim_qk));
        variant += std::string("_hsqk") + std::to_string(key.head_dim_qk);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(key.head_dim_v));
        variant += std::string("_hsv") + std::to_string(key.head_dim_v);

        const char * shader_src = wgsl_flash_attn;
        if (key.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
            defines.push_back("KV_GRANULARITY=8");
            defines.push_back(std::string("VEC_NE=") + std::to_string(ggml_webgpu_flash_attn_pick_vec_ne(key)) + "u");
            shader_src = wgsl_flash_attn_vec_split;
        } else if (key.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
            shader_src = wgsl_flash_attn_tile;
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size));
            defines.push_back("KV_STAGE_STRIDE=" + std::to_string(std::max(key.head_dim_qk, key.head_dim_v)));
            variant += "_tile";
        } else {
            defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
            defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
            defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
        }

        auto pipeline_decisions        = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
        pipeline_decisions->kv_overlap = key.kv_overlap;
        defines.push_back(std::string("Q_TILE=") + std::to_string(decisions.q_tile));
        defines.push_back(std::string("KV_TILE=") + std::to_string(decisions.kv_tile));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(decisions.wg_size));

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
        pipeline.context          = pipeline_decisions;
        flash_attn_pipelines[key] = pipeline;
        return flash_attn_pipelines[key];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(const ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
        ggml_webgpu_flash_attn_blk_pipeline_key key = {};
        key.kv_tile                                 = kv_tile;
        auto it                                     = flash_attn_blk_pipelines.find(key);
        if (it != flash_attn_blk_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(key.kv_tile));
        variant += std::string("_kvt") + std::to_string(key.kv_tile);

        uint32_t wg_size = 1;
        while ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[key] = pipeline;
        return flash_attn_blk_pipelines[key];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_key key = {};
        key.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        key.wg_size                                        = context.max_wg_size;
        auto it                                            = flash_attn_vec_reduce_pipelines.find(key);
        if (it != flash_attn_vec_reduce_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(key.head_dim_v));
        variant += std::string("_hsv") + std::to_string(key.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[key] = pipeline;
        return flash_attn_vec_reduce_pipelines[key];
    }

    webgpu_pipeline get_cpy_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_key key = {};
        key.src_type                     = context.src0->type;
        key.dst_type                     = context.dst->type;

        auto it = cpy_pipelines.find(key);
        if (it != cpy_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        switch (key.src_type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                break;
            default:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        switch (key.dst_type) {
            case GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                break;
            case GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                break;
            default:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_cpy, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[key]       = pipeline;
        return cpy_pipelines[key];
    }

    webgpu_pipeline get_glu_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_glu_pipeline_key key = {};
        key.glu_op                       = ggml_get_glu_op(context.dst);
        key.type                         = context.dst->type;
        key.split                        = (context.src1 != nullptr);

        auto it = glu_pipelines.find(key);
        if (it != glu_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        switch (key.glu_op) {
            case GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                break;
            case GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                break;
            case GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                break;
            case GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                break;
            case GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                break;
            case GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                break;
            default:
                GGML_ABORT("Unsupported GLU op");
        }
        switch (key.type) {
            case GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                break;
            default:
                GGML_ABORT("Unsupported type for GLU shader");
        }

        if (key.split) {
            variant += "_split";
        } else {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_glu, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[key]       = pipeline;
        return glu_pipelines[key];
    }

    webgpu_pipeline get_rope_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_key key = {};
        key.type                          = context.dst->type;
        key.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
        key.has_ff                        = (context.src2 != nullptr);

        auto it = rope_pipelines.find(key);
        if (it != rope_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

        switch (key.type) {
            case GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("TYPE_F16");
                variant += "_f16";
                break;
            default:
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        if (key.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_rope, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = key.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[key]      = pipeline;
        return rope_pipelines[key];
    }

    webgpu_pipeline get_soft_max_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_key key = {};
        key.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        key.has_mask                          = (context.src1 != nullptr);
        key.has_sink                          = (context.src2 != nullptr);
        key.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);

        auto it = soft_max_pipelines.find(key);
        if (it != soft_max_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        if (key.has_mask) {
            defines.push_back("HAS_MASK");
            switch (key.mask_type) {
                case GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    break;
                case GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    break;
                default:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        if (key.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        decisions->inplace       = key.inplace;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[key]  = pipeline;
        return soft_max_pipelines[key];
    }

    webgpu_pipeline get_conv2d_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_key key = {};
        key.weight_type                     = context.src0->type;
        key.input_type                      = context.src1->type;
        key.output_type                     = context.dst->type;

        auto it = conv2d_pipelines.find(key);
        if (it != conv2d_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        auto push_type_defines = [&](const char * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            if (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "_F32");
            } else if (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "_F16");
            } else {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        push_type_defines("WEIGHT", key.weight_type);
        push_type_defines("INPUT", key.input_type);
        push_type_defines("OUTPUT", key.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[key]    = pipeline;
        return conv2d_pipelines[key];
    }

    webgpu_pipeline get_im2col_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_key key = {};
        key.input_type                      = context.src1->type;
        key.output_type                     = context.dst->type;

        auto it = im2col_pipelines.find(key);
        if (it != im2col_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        auto push_type_defines = [&](const char * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            if (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "_F32");
            } else if (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "_F16");
            } else {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        push_type_defines("INPUT", key.input_type);
        push_type_defines("OUTPUT", key.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[key]    = pipeline;
        return im2col_pipelines[key];
    }

    webgpu_pipeline get_upscale_pipeline(const ggml_webgpu_shader_lib_context & context) {
        const uint32_t mode_flags = (uint32_t) ggml_get_op_params_i32(context.dst, 0);
        const uint32_t base_mode  = mode_flags & 0xFFu;
        const bool     antialias  = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0u;

        ggml_webgpu_upscale_pipeline_key key = {};
        key.input_type                       = context.src0->type;
        key.output_type                      = context.dst->type;
        key.base_mode                        = base_mode;
        key.antialias                        = antialias;

        auto it = upscale_pipelines.find(key);
        if (it != upscale_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "upscale";

        if (key.input_type == GGML_TYPE_F16) {
            defines.push_back("SRC_F16");
            variant += "_src_f16";
        } else {
            variant += "_src_f32";
        }

        if (key.output_type == GGML_TYPE_F16) {
            defines.push_back("DST_F16");
            variant += "_dst_f16";
        } else {
            variant += "_dst_f32";
        }

        switch (base_mode) {
            case GGML_SCALE_MODE_NEAREST:
                defines.push_back("NEAREST");
                variant += "_nearest";
                break;
            case GGML_SCALE_MODE_BILINEAR:
                defines.push_back("BILINEAR");
                variant += "_bilinear";
                break;
            case GGML_SCALE_MODE_BICUBIC:
                defines.push_back("BICUBIC");
                variant += "_bicubic";
                break;
            default:
                GGML_ABORT("Unsupported upscale mode");
        }

        if (antialias) {
            defines.push_back("ANTIALIAS");
            variant += "_aa";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_upscale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        upscale_pipelines[key]   = pipeline;
        return upscale_pipelines[key];
    }

  private:
    static webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_code,
                                                       std::string    label) {
        wgpu::ShaderSourceWGSL shader_source;
        shader_source.code = shader_code.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.nextInChain = &shader_source;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = nullptr;  // nullptr means auto layout
        return { device.CreateComputePipeline(&pipeline_desc), label };
    }
};


#include "ggml.h.inc"

#ifdef __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#ifdef GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#endif
#if defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#endif
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#define CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
static inline void compute_2d_workgroups(uint32_t total_wg, uint32_t max_per_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(total_wg, max_per_dim));
    wg_x = CEIL_DIV(total_wg, wg_y);
}

static inline uint32_t ggml_webgpu_u32_from_f32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

#ifdef GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(msg)  std::cout << msg << std::endl
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#else
#    define WEBGPU_LOG_DEBUG(msg) ((void) 0)
#endif  // GGML_WEBGPU_DEBUG

#ifdef GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_TOTAL_START(id) auto cpu_total_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)                                                         \
        auto   cpu_total_end_##id = std::chrono::high_resolution_clock::now();                            \
        double cpu_total_time_##id =                                                                      \
            std::chrono::duration<double, std::milli>(cpu_total_end_##id - cpu_total_start_##id).count(); \
        (ctx)->cpu_time_ms[#id] += cpu_total_time_##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id) auto cpu_detail_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)                                                          \
        auto   cpu_detail_end_##id = std::chrono::high_resolution_clock::now();                             \
        double cpu_detail_time_##id =                                                                       \
            std::chrono::duration<double, std::milli>(cpu_detail_end_##id - cpu_detail_start_##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_time_##id;
#else
#    define WEBGPU_CPU_PROFILE_TOTAL_START(id)
#    define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)
#endif  // GGML_WEBGPU_CPU_PROFILE

#ifdef GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_COUNT        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES (WEBGPU_MAX_PROFILE_QUERY_COUNT * sizeof(uint64_t))
#endif

/* Constants */

#define WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE 64u
#define WEBGPU_NUM_PARAM_SLOT_SAFETY_MARGIN      10u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#define WEBGPU_PARAMS_BUF_SIZE_BYTES             128  // enough for 32 parameters
#define WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES     4
#define WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

// For operations which process a row in parallel, this seems like a reasonable
// default
#define WEBGPU_ROW_SPLIT_WG_SIZE 64

// Track https://github.com/gpuweb/gpuweb/issues/5315 for fixes to
// implementations so this can be removed, necessary only for get_rows right now
#define WEBGPU_MAX_WG_SIZE 288

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
static void * const webgpu_ptr_base = (void *) (uintptr_t) 0x1000;  // NOLINT

static size_t ggml_webgpu_tensor_offset(const ggml_tensor * tensor) {
    const ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    return (size_t) ((uintptr_t) base_tensor->data - (uintptr_t) webgpu_ptr_base) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
static void ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      const char *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
struct webgpu_param_arena {
    wgpu::Buffer buffer;
    size_t       slot_stride = 0;
    size_t       slot_size   = 0;
    uint32_t     slot_count  = 0;
    uint32_t     next_slot   = 0;

    void init(wgpu::Device device, size_t slot_size, uint32_t slot_count, size_t alignment) {
        this->slot_stride = ROUNDUP_POW2(slot_size, alignment);
        this->slot_size   = slot_size;
        this->slot_count  = slot_count;
        this->next_slot   = 0;

        ggml_webgpu_create_buffer(device, buffer, this->slot_stride * slot_count,
                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_slot(size_t size) {
        GGML_ASSERT(size <= slot_size);
        if (next_slot >= slot_count) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        return slot_stride * next_slot++;
    }

    void reset() { next_slot = 0; }

    void cleanup() {
        if (buffer) {
            buffer.Destroy();
            buffer = nullptr;
        }
    }

    ~webgpu_param_arena() { this->cleanup(); }
};

struct webgpu_encoded_op {
    uint32_t num_kernels = 0;
#ifdef GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_names;
#endif
};

struct webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> bind_group_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

struct webgpu_capabilities {
    wgpu::Limits limits;
    bool         supports_subgroups       = false;
    bool         supports_subgroup_matrix = false;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_size     = 0;
    uint32_t max_subgroup_size = 0;
    size_t   memset_bytes_per_thread;
};

// Stores global webgpu members
struct webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::Queue    queue;
    uint32_t       command_submit_batch_size = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::Buffer         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::Buffer    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#ifdef GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, double> cpu_time_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, double> cpu_detail_ms;
#endif

#ifdef GGML_WEBGPU_DEBUG
    wgpu::Buffer debug_host_buf;
    wgpu::Buffer debug_dev_buf;
#endif

    ~webgpu_global_context_struct() {
        if (this->get_tensor_staging_buf) {
            this->get_tensor_staging_buf.Destroy();
            this->get_tensor_staging_buf = nullptr;
        }
        if (this->memset_params_buf) {
            this->memset_params_buf.Destroy();
            this->memset_params_buf = nullptr;
        }
#ifdef GGML_WEBGPU_DEBUG
        if (this->debug_host_buf) {
            this->debug_host_buf.Destroy();
            this->debug_host_buf = nullptr;
        }
        if (this->debug_dev_buf) {
            this->debug_dev_buf.Destroy();
            this->debug_dev_buf = nullptr;
        }
#endif
    }
};

typedef std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
struct webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::Buffer             set_rows_dev_error_buf;
    wgpu::Buffer             set_rows_host_error_buf;
    wgpu::CommandEncoder     active_command_encoder;
    wgpu::ComputePassEncoder active_compute_pass;

    size_t memset_bytes_per_thread;

#ifdef GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, double> shader_gpu_time_ms;
    wgpu::Buffer                            profile_timestamp_dev_buf;
    wgpu::Buffer                            profile_timestamp_host_buf;
    wgpu::QuerySet                          profile_timestamp_query_set;
    uint32_t                                profile_timestamp_query_count = 0;
#endif

    ~webgpu_context_struct() {
#ifdef GGML_WEBGPU_GPU_PROFILE
        if (this->profile_timestamp_host_buf) {
            this->profile_timestamp_host_buf.Destroy();
            this->profile_timestamp_host_buf = nullptr;
        }
        if (this->profile_timestamp_dev_buf) {
            this->profile_timestamp_dev_buf.Destroy();
            this->profile_timestamp_dev_buf = nullptr;
        }
        if (this->profile_timestamp_query_set) {
            this->profile_timestamp_query_set.Destroy();
            this->profile_timestamp_query_set = nullptr;
        }
#endif
        if (this->set_rows_host_error_buf) {
            this->set_rows_host_error_buf.Destroy();
            this->set_rows_host_error_buf = nullptr;
        }
        if (this->set_rows_dev_error_buf) {
            this->set_rows_dev_error_buf.Destroy();
            this->set_rows_dev_error_buf = nullptr;
        }
    }
};

typedef std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
struct ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_count;
    const char *          name;
};

// Per-device struct for the global logical device interface
struct ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
struct ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
struct ggml_backend_webgpu_buffer_context {
    wgpu::Buffer          buffer;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_buffer_context(wgpu::Buffer buf, std::string lbl, webgpu_global_context global_ctx_) :
        buffer(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

static webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   const char *                             shader_code,
                                                   const char *                             label,
                                                   const std::vector<wgpu::ConstantEntry> & constants = {}) {
    wgpu::ShaderSourceWGSL shader_source;
    shader_source.code = shader_code;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.nextInChain = &shader_source;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = nullptr;  // nullptr means auto layout
    if (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.data();
        pipeline_desc.compute.constantCount = constants.size();
    }
    return { device.CreateComputePipeline(&pipeline_desc), label };
}

static void ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      const char *      label) {
    wgpu::BufferDescriptor buffer_desc;
    buffer_desc.size             = size;
    buffer_desc.usage            = usage;
    buffer_desc.label            = label;
    buffer_desc.mappedAtCreation = false;

    // TODO: error handling
    buffer = device.CreateBuffer(&buffer_desc);
}

static wgpu::Buffer ggml_webgpu_tensor_buf(const ggml_tensor * tensor) {
    ggml_backend_webgpu_buffer_context * ctx = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    return ctx->buffer;
}

static size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, const ggml_tensor * t) {
    size_t offset = ggml_webgpu_tensor_offset(t);
    return offset & (ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment - 1);
}

static size_t ggml_webgpu_tensor_align_offset(webgpu_context & ctx, const ggml_tensor * t) {
    size_t offset = ggml_webgpu_tensor_offset(t);
    return offset & ~(ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment - 1);
}

static size_t ggml_webgpu_tensor_binding_size(webgpu_context & ctx, ggml_tensor * t) {
    return ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(ctx, t), WEBGPU_STORAGE_BUF_BINDING_MULT);
}

struct ggml_webgpu_merged_binding_range {
    size_t offset;
    size_t size;
};

static ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_offset = SIZE_MAX;
    size_t merged_end    = 0;

    for (ggml_tensor * tensor : tensors) {
        const size_t bind_offset = ggml_webgpu_tensor_align_offset(ctx, tensor);
        const size_t bind_end    = bind_offset + ggml_webgpu_tensor_binding_size(ctx, tensor);

        merged_offset = std::min(merged_offset, bind_offset);
        merged_end    = std::max(merged_end, bind_end);
    }

    return { merged_offset, merged_end - merged_offset };
}

static uint32_t ggml_webgpu_tensor_merged_element_offset(const ggml_tensor *                      tensor,
                                                         const ggml_webgpu_merged_binding_range & merged_range) {
    return (uint32_t) ((ggml_webgpu_tensor_offset(tensor) - merged_range.offset) / ggml_type_size(tensor->type));
}

static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t     binding,
                                                              wgpu::Buffer buffer,
                                                              uint64_t     offset,
                                                              uint64_t     size) {
    wgpu::BindGroupEntry entry = {};
    entry.binding              = binding;
    entry.buffer               = std::move(buffer);
    entry.offset               = offset;
    entry.size                 = size;
    return entry;
}

static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    return ggml_webgpu_make_bind_group_entry(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_offset(ctx, tensor),
                                             ggml_webgpu_tensor_binding_size(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

template <typename T>
static void ggml_backend_webgpu_check_wait_status(wgpu::WaitStatus wait_status,
                                                  T                callback_status,
                                                  T                success_status,
                                                  const char *     wait_name,
                                                  const char *     failure_name,
                                                  const char *     callback_message) {
    if (wait_status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    if (wait_status == wgpu::WaitStatus::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    if (callback_status != success_status) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<int>(callback_status),
                   callback_message);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
static uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    return UINT32_MAX;
}

static uint32_t ggml_backend_webgpu_get_command_submit_batch_size() {
    return WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE;
}

static void ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) {
    wgpu::QueueWorkDoneStatus callback_status = wgpu::QueueWorkDoneStatus::Error;
    std::string               callback_message;

    const wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        ctx->queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::AllowSpontaneous,
            [&callback_status, &callback_message](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                callback_status  = status;
                callback_message = std::string(message);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::QueueWorkDoneStatus::Success,
                                          "Queue wait", "Queue work", callback_message.c_str());
}

static void ggml_backend_webgpu_map_buffer(webgpu_global_context & ctx,
                                           wgpu::Buffer &          buffer,
                                           wgpu::MapMode           mode,
                                           size_t                  offset,
                                           size_t                  size) {
    wgpu::MapAsyncStatus callback_status = wgpu::MapAsyncStatus::Error;
    std::string          callback_message;

    const wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        buffer.MapAsync(mode, offset, size, wgpu::CallbackMode::AllowSpontaneous,
                        [&callback_status, &callback_message](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                            callback_status  = status;
                            callback_message = std::string(message);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::MapAsyncStatus::Success,
                                          "Buffer map wait", "Buffer map", callback_message.c_str());
}

static void ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                const wgpu::CommandBuffer commands,
                                                uint32_t &                num_inflight_batches) {
    if (num_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_queue(ctx->global_ctx);
        num_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    num_inflight_batches++;
}

#ifdef GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
static void ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_buffer(ctx, ctx->debug_host_buf, wgpu::MapMode::Read, 0, ctx->debug_host_buf.GetSize());
    const float * debug_data = (const float *) ctx->debug_host_buf.GetConstMappedRange();
    std::cout << "debug[0]: " << debug_data[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#endif

static webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         const std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            result = {};
    std::vector<wgpu::BindGroup> bind_groups;
    std::vector<size_t>          param_offsets;
    result.num_kernels = dispatches.size();

    for (size_t i = 0; i < dispatches.size(); i++) {
        const webgpu_dispatch_desc & dispatch     = dispatches[i];
        const size_t                 param_size   = dispatch.params.size() * sizeof(uint32_t);
        const size_t                 param_offset = ctx->param_arena.alloc_slot(param_size);

        std::vector<wgpu::BindGroupEntry> entries            = dispatch.bind_group_entries;
        uint32_t                          params_binding_num = entries.size();
        entries.push_back(ggml_webgpu_make_bind_group_entry(params_binding_num, ctx->param_arena.buffer, param_offset,
                                                            ctx->param_arena.slot_size));

        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout     = dispatch.pipeline.pipeline.GetBindGroupLayout(0);
        bind_group_desc.entryCount = entries.size();
        bind_group_desc.entries    = entries.data();
        bind_group_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.CreateBindGroup(&bind_group_desc));
        param_offsets.push_back(param_offset);
    }

    for (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.WriteBuffer(ctx->param_arena.buffer, param_offsets[i], dispatches[i].params.data(),
                                           dispatches[i].params.size() * sizeof(uint32_t));
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    for (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_count + 2 <= WEBGPU_MAX_PROFILE_QUERY_COUNT);
        const uint32_t query_begin = ctx->profile_timestamp_query_count++;
        const uint32_t query_end   = ctx->profile_timestamp_query_count++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.querySet                    = ctx->profile_timestamp_query_set;
        ts_writes.beginningOfPassWriteIndex   = query_begin;
        ts_writes.endOfPassWriteIndex         = query_end;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass(&pass_desc);

        pass.SetPipeline(dispatches[i].pipeline.pipeline);
        pass.SetBindGroup(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
        pass.End();
        result.pipeline_names.push_back(dispatches[i].pipeline.name);
    }
#else
    for (size_t i = 0; i < dispatches.size(); i++) {
        ctx->active_compute_pass.SetPipeline(dispatches[i].pipeline.pipeline);
        ctx->active_compute_pass.SetBindGroup(0, bind_groups[i]);
        ctx->active_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
    }
#endif

    return result;
}

static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::BindGroupEntry> bind_group_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    return ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_group_entries), { wg_x, wg_y } },
    });
}

static void ggml_backend_webgpu_buffer_memset(webgpu_global_context & ctx,
                                              wgpu::Buffer &          buf,
                                              uint32_t                value,
                                              size_t                  offset,
                                              size_t                  size) {
    std::vector<uint32_t>             params       = { (uint32_t) offset, (uint32_t) size, value };
    std::vector<wgpu::BindGroupEntry> entries      = { ggml_webgpu_make_bind_group_entry(0, buf, 0, buf.GetSize()) };
    size_t                            bytes_per_wg = WEBGPU_MAX_WG_SIZE * ctx->capabilities.memset_bytes_per_thread;
    uint32_t                          wg_x         = CEIL_DIV(size + 3, bytes_per_wg);

    ctx->queue.WriteBuffer(ctx->memset_params_buf, 0, params.data(), params.size() * sizeof(uint32_t));

    wgpu::BindGroupEntry params_entry = {};
    params_entry.binding              = 1;
    params_entry.buffer               = ctx->memset_params_buf;
    params_entry.offset               = 0;
    params_entry.size                 = WEBGPU_PARAMS_BUF_SIZE_BYTES;
    entries.push_back(params_entry);

    wgpu::BindGroupDescriptor bind_group_desc;
    bind_group_desc.layout     = ctx->memset_pipeline.pipeline.GetBindGroupLayout(0);
    bind_group_desc.entryCount = entries.size();
    bind_group_desc.entries    = entries.data();
    bind_group_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.SetPipeline(ctx->memset_pipeline.pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.End();

    wgpu::CommandBuffer              command  = encoder.Finish();
    std::vector<wgpu::CommandBuffer> commands = { command };
    ctx->queue.Submit(commands.size(), commands.data());
}

/** End WebGPU Actions */

/** GGML Backend Interface */

static const char * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#ifdef GGML_WEBGPU_CPU_PROFILE
    std::cout << "\n[ggml_webgpu cpu profiling summary]\n";
    double total_cpu = 0.0;
    for (const auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        total_cpu += kv.second;
    }
    std::cout << "ggml_webgpu: total cpu time: " << total_cpu << " ms\n";
    std::cout << "ggml_webgpu: cpu breakdown:\n";
    for (const auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        double pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    if (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::cout << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    for (const auto & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        double pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#endif

#ifdef GGML_WEBGPU_GPU_PROFILE
    std::cout << "\n[ggml_webgpu gpu profiling summary]\n";
    double total_gpu = 0.0;
    for (const auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        total_gpu += kv.second;
    }
    std::cout << "ggml_webgpu: total gpu time (all shaders): " << total_gpu << " ms\n";
    std::cout << "\nggml_webgpu: gpu breakdown:\n";
    for (const auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        double pct = (total_gpu > 0.0) ? (kv.second / total_gpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#endif

#if defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::cout << "ggml_webgpu: gpu/cpu ratio: " << (total_cpu > 0.0 ? total_gpu / total_cpu : 0.0) << "\n";
#endif

    delete ctx;
    delete backend;
}

static webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_size(src->type)), (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)), (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        // Logical shapes
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1], (uint32_t) dst->ne[2]
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_set(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    const bool inplace   = decisions->inplace;

    const uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    const uint32_t dst_type_size = (uint32_t) ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (((const int32_t *) dst->op_params)[3] / dst_type_size),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        1u,
        (uint32_t) (((const int32_t *) dst->op_params)[0] / dst_type_size),
        (uint32_t) (((const int32_t *) dst->op_params)[1] / dst_type_size),
        (uint32_t) (((const int32_t *) dst->op_params)[2] / dst_type_size),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t                          binding_index = 0;
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        binding_index++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index + 1, dst));

    uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        // Shapes
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
        // Pad sizes
        (uint32_t) ggml_get_op_params_i32(dst, 0),
        (uint32_t) ggml_get_op_params_i32(dst, 1),
        (uint32_t) ggml_get_op_params_i32(dst, 2),
        (uint32_t) ggml_get_op_params_i32(dst, 3),
        (uint32_t) ggml_get_op_params_i32(dst, 4),
        (uint32_t) ggml_get_op_params_i32(dst, 5),
        (uint32_t) ggml_get_op_params_i32(dst, 6),
        (uint32_t) ggml_get_op_params_i32(dst, 7),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    const uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_size);
    const uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t p0 = ggml_get_op_params_i32(dst, 2);
    const int32_t p1 = ggml_get_op_params_i32(dst, 3);
    const int32_t d0 = ggml_get_op_params_i32(dst, 4);
    const int32_t d1 = ggml_get_op_params_i32(dst, 5);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],

        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],

        (uint32_t) s0,
        (uint32_t) s1,
        (uint32_t) p0,
        (uint32_t) p1,
        (uint32_t) d0,
        (uint32_t) d1,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    uint32_t wg_x     = std::min(ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, total_wg);
    uint32_t wg_y     = CEIL_DIV(total_wg, wg_x);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    const int32_t s0    = ggml_get_op_params_i32(dst, 0);
    const int32_t s1    = ggml_get_op_params_i32(dst, 1);
    const int32_t p0    = ggml_get_op_params_i32(dst, 2);
    const int32_t p1    = ggml_get_op_params_i32(dst, 3);
    const int32_t d0    = ggml_get_op_params_i32(dst, 4);
    const int32_t d1    = ggml_get_op_params_i32(dst, 5);
    const bool    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    const uint32_t KW = src0->ne[0];
    const uint32_t KH = is_2D ? src0->ne[1] : 1;
    const uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    const uint32_t IW = src1->ne[0];
    const uint32_t IH = is_2D ? src1->ne[1] : 1;
    const uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    const uint32_t OW = dst->ne[1];
    const uint32_t OH = is_2D ? dst->ne[2] : 1;

    const uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_size(src1->type));
    const uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0;
    const uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_size(src1->type));
    const uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_size(src1->type));

    const uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_size(dst->type));
    const uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_size(dst->type));
    const uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)) : 0;
    const uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_size(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        si0,
        si1,
        si2,
        si3,
        so0,
        so1,
        so2,
        so3,

        KW,
        KH,
        IC,

        IW,
        IH,
        N,

        OW,
        OH,

        (uint32_t) s0,
        (uint32_t) s1,
        (uint32_t) p0,
        (uint32_t) p1,
        (uint32_t) d0,
        (uint32_t) d1,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    uint32_t wg_x     = std::min(ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, total_wg);
    uint32_t wg_y     = CEIL_DIV(total_wg, wg_x);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    const uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->tokens_per_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        token_tiles,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    const uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->block_size);
    const uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_ssm_scan(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    src3,
                                              ggml_tensor *    src4,
                                              ggml_tensor *    src5,
                                              ggml_tensor *    src6,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src4                           = src4;
    shader_lib_ctx.src5                           = src5;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline pipeline    = ctx->shader_lib->get_ssm_scan_pipeline(shader_lib_ctx);
    auto *          decisions   = static_cast<ggml_webgpu_ssm_scan_shader_decisions *>(pipeline.context.get());
    const bool      xbc_overlap = decisions->xbc_overlap;

    uint32_t offset_x        = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    uint32_t offset_B        = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_size(src4->type));
    uint32_t offset_C        = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_size(src5->type));
    size_t   xbc_bind_offset = 0;
    size_t   xbc_bind_size   = 0;
    if (xbc_overlap) {
        const ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        xbc_bind_offset = merged_range.offset;
        xbc_bind_size   = merged_range.size;
        offset_x        = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
        offset_B        = ggml_webgpu_tensor_merged_element_offset(src4, merged_range);
        offset_C        = ggml_webgpu_tensor_merged_element_offset(src5, merged_range);
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        offset_x,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_size(src3->type)),
        offset_B,
        offset_C,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_size(src6->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_size(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_size(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_size(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_size(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_size(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_size(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
        (uint32_t) ggml_nelements(src1),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
    };
    if (xbc_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), xbc_bind_offset, xbc_bind_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src6));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, dst));
    } else {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src4));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, src5));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 6, src6));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 7, dst));
    }

    const uint32_t total_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
                                                     ggml_tensor *    src0,
                                                     ggml_tensor *    src1,
                                                     ggml_tensor *    src2,
                                                     ggml_tensor *    src3,
                                                     ggml_tensor *    src4,
                                                     ggml_tensor *    src5,
                                                     ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.src3                           = src3;
    shader_lib_ctx.src4                           = src4;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    const uint32_t s_v      = (uint32_t) src2->ne[0];
    const uint32_t h        = (uint32_t) src2->ne[1];
    const uint32_t n_tokens = (uint32_t) src2->ne[2];
    const uint32_t n_seqs   = (uint32_t) src2->ne[3];
    const float    scale    = 1.0f / sqrtf((float) s_v);
    uint32_t       scale_u32;
    memcpy(&scale_u32, &scale, sizeof(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_tokens,
        n_seqs,
        s_v * h * n_tokens * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_size(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_size(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        scale_u32,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0), ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2), ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src4), ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 6, dst),
    };

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

static std::optional<webgpu_encoded_op> ggml_webgpu_set_rows(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    if (ggml_is_empty(src) || ggml_is_empty(idx)) {
        return std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_rows_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_set_rows_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_size(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)), (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)), (uint32_t) (idx->nb[0] / ggml_type_size(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_size(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_size(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, idx),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    if (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(3, ctx->set_rows_dev_error_buf, 0,
                                                            ctx->set_rows_dev_error_buf.GetSize()));
    }

    uint32_t threads;
    if (decisions->vec4) {
        threads = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } else {
        threads = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x = CEIL_DIV(threads, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, 1);
}

// Workgroup size is a common constant
static std::vector<wgpu::ConstantEntry> ggml_webgpu_wg_size_entry(uint32_t wg_size) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].key   = "wg_size";
    constants[0].value = wg_size;
    return constants;
}

static webgpu_encoded_op ggml_webgpu_get_rows(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    const bool float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size                    = WEBGPU_MAX_WG_SIZE;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_rows_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_size(idx->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     (uint32_t) (idx->nb[0] / ggml_type_size(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_size(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_size(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],
                                     (uint32_t) (idx->ne[1]),
                                     (uint32_t) (idx->ne[2]) };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst) };

    uint32_t blocks_per_row = (uint32_t) (dst->ne[0] / (src->type == GGML_TYPE_F32 && dst->ne[0] % 4 == 0 ? 4 : 1));
    uint32_t total_rows     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t total_threads  = float_parallel ? blocks_per_row * total_rows : total_rows;
    uint32_t wg_x           = CEIL_DIV(total_threads, decisions->wg_size);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    bool is_vec = (dst->ne[1] == 1);

    // Determine if we should use fast path
    bool use_fast = false;
    switch (src1->type) {
        case GGML_TYPE_F16:
            use_fast = (src0->type == GGML_TYPE_F16);
            break;
        case GGML_TYPE_F32:
            // TODO: implement better mat-mat for k-quants, mat-vec for all k-quants except q6_K
            switch (src0->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q1_0:
                    use_fast = true;
                    break;
                case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ1_M:
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    use_fast = true;
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_size              = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.max_subgroup_size        = ctx->global_ctx->capabilities.max_subgroup_size;

    // Get or create pipeline
    webgpu_pipeline pipeline;

    if (use_fast && is_vec) {
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx);
    } else if (use_fast) {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx);
    } else {
        pipeline = ctx->shader_lib->get_mul_mat_legacy_pipeline(shader_lib_ctx);
    }

    // Build params
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) src0->ne[0],
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
        (uint32_t) src0->ne[2],
        (uint32_t) src0->ne[3],
        (uint32_t) (src1->ne[2] / src0->ne[2]),
        (uint32_t) (src1->ne[3] / src0->ne[3])
    };

    // Build bind group entries
    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;

    if (use_fast && is_vec) {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
        uint32_t total_wg      = output_groups * batches;
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
    } else if (use_fast) {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        if (decisions->use_subgroup_matrix) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_matrix_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_matrix_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } else {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_size_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_size_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t total_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    } else {  // legacy
        auto *   decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
        uint32_t wg_size   = decisions->wg_size;
        uint32_t total_wg  = CEIL_DIV(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3], wg_size);
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
    }

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    const uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    const uint32_t param_n_expert_used = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_used,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_size(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_size(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_size(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_size(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
    uint32_t       total_wg       = output_groups * param_n_expert_used;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    if (dst->ne[2] == 1) {
        return ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    const uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    const uint32_t param_n_expert_used = (uint32_t) dst->ne[1];
    const uint32_t param_n_tokens      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
    };

    const size_t dst_offset          = ggml_webgpu_tensor_offset(dst);
    const size_t gathered_buf_nbytes = src0->ne[2] * src1->ne[2] * sizeof(uint32_t);

    const size_t gathered_expert_used_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    const size_t gathered_tokens_align_offset =
        ROUNDUP_POW2(gathered_expert_used_align_offset + gathered_buf_nbytes,
                     ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    const size_t gathered_count_ids_align_offset =
        ROUNDUP_POW2(gathered_tokens_align_offset + gathered_buf_nbytes,
                     ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);

    const size_t gathered_binding_size = ROUNDUP_POW2(gathered_buf_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    const size_t gathered_count_ids_binding_size =
        ROUNDUP_POW2(src0->ne[2] * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::BindGroupEntry> gather_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_size(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), gathered_expert_used_align_offset,
                                          gathered_binding_size),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), gathered_tokens_align_offset,
                                          gathered_binding_size),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), gathered_count_ids_align_offset,
                                          gathered_count_ids_binding_size),
    };

    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;

    const uint32_t gather_total_wg = param_n_expert;
    const uint32_t gather_wg_x     = std::min(gather_total_wg, max_wg_per_dim);
    const uint32_t gather_wg_y     = CEIL_DIV(gather_total_wg, gather_wg_x);

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { gather_wg_x, gather_wg_y }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::BindGroupEntry> main_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_size(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_size(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_size(ctx, dst)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), gathered_expert_used_align_offset,
                                          gathered_binding_size),
        ggml_webgpu_make_bind_group_entry(4, ggml_webgpu_tensor_buf(dst), gathered_tokens_align_offset,
                                          gathered_binding_size),
        ggml_webgpu_make_bind_group_entry(5, ggml_webgpu_tensor_buf(dst), gathered_count_ids_align_offset,
                                          gathered_count_ids_binding_size),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    auto * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    uint32_t wg_m;

    uint32_t tile_m_s           = main_decisions->tile_m * main_decisions->wg_size_m;
    uint32_t tile_n_s           = main_decisions->tile_n * main_decisions->wg_size_n;
    wg_m                        = CEIL_DIV(dst->ne[0], tile_m_s);
    uint32_t total_gathered     = dst->ne[1] * dst->ne[2];
    uint32_t max_active_experts = std::min((uint32_t) src0->ne[2], total_gathered);
    uint32_t max_wg_n           = CEIL_DIV(total_gathered, tile_n_s) + max_active_experts;
    uint32_t total_wg           = wg_m * max_wg_n;

    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    mask,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    float scale         = ggml_get_op_params_f32(dst, 0);
    float max_bias      = ggml_get_op_params_f32(dst, 1);
    float logit_softcap = ggml_get_op_params_f32(dst, 2);
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    float n_head_log2 = float(1u << (uint32_t) floor(log2(Q->ne[2])));
    float m0          = powf(2.0f, -(max_bias) / n_head_log2);
    float m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = Q;
    shader_lib_ctx.src1                           = K;
    shader_lib_ctx.src2                           = V;
    shader_lib_ctx.src3                           = mask;
    shader_lib_ctx.src4                           = sinks;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_matrix       = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
    shader_lib_ctx.sg_mat_m           = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n           = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k           = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.max_subgroup_size  = ctx->global_ctx->capabilities.max_subgroup_size;
    webgpu_pipeline pipeline          = ctx->shader_lib->get_flash_attn_pipeline(
        shader_lib_ctx, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    auto *     decisions  = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    const int  has_mask   = (mask != nullptr);
    const int  has_sinks  = (sinks != nullptr);
    const bool kv_overlap = decisions->kv_overlap;

    uint32_t offset_k       = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type));
    uint32_t offset_v       = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type));
    size_t   kv_bind_offset = 0;
    size_t   kv_bind_size   = 0;
    if (kv_overlap) {
        const ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        kv_bind_offset                                      = merged_range.offset;
        kv_bind_size                                        = merged_range.size;
        offset_k                                            = ggml_webgpu_tensor_merged_element_offset(K, merged_range);
        offset_v                                            = ggml_webgpu_tensor_merged_element_offset(V, merged_range);
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_size(Q->type)),
        offset_k,
        offset_v,
        has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)) : 0,
        has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_size(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 3
        has_mask ? (uint32_t) (mask->nb[3] / ggml_type_size(mask->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_head_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)

    };
    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, Q),
    };
    if (kv_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), kv_bind_offset, kv_bind_size));
    } else {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V));
    }
    uint32_t binding_index = kv_overlap ? 2u : 3u;
    if (has_mask) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, mask));
    }
    if (has_sinks) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, sinks));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, dst));

    if (decisions->path != GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
        uint32_t wg_per_head = CEIL_DIV(Q->ne[1], decisions->q_tile);
        uint32_t wg_x        = wg_per_head * Q->ne[2] * Q->ne[3];  // wg per head * number of heads * number of batches
        return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
    }

    wgpu::Buffer blk_buf         = {};
    uint64_t     blk_size_bytes  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_count = 0;

    const uint32_t vec_nwg_cap = std::max(1u, std::min<uint32_t>(32u, ctx->global_ctx->capabilities.max_subgroup_size));
    uint32_t       nwg         = 1u;
    const uint64_t kv_span     = (uint64_t) std::max(1u, decisions->kv_tile);
    while ((2u * nwg * kv_span) < (uint64_t) K->ne[1] && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    nwg                           = std::min(nwg, vec_nwg_cap);
    const uint64_t nrows          = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    const bool     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_base  = 0;
    uint64_t     tmp_size_bytes  = 0;
    wgpu::Buffer tmp_buf         = {};
    uint64_t     tmp_bind_offset = 0;
    uint64_t     tmp_bind_size   = 0;
    const size_t align_bytes     = ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    const size_t dst_offset      = ggml_webgpu_tensor_offset(dst);
    size_t       scratch_offset  = ROUNDUP_POW2(dst_offset + ggml_nbytes(dst), align_bytes);

    if (use_vec_reduce) {
        const uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        const uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_base                 = tmp_data_elems;
        tmp_size_bytes =
            ROUNDUP_POW2((tmp_data_elems + tmp_stats_elems) * sizeof(float), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_base <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_size   = tmp_size_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
    } else {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_size_bytes  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_size   = tmp_size_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::BindGroupEntry> blk_entries;
    if (has_mask) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        const uint32_t stride_mask3 = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
        blk_batch_count             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        const uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_count;
        blk_size_bytes              = ROUNDUP_POW2(blk_elems * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        const ggml_webgpu_shader_lib_context blk_shader_ctx = shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(mask),
                                              ggml_webgpu_tensor_align_offset(ctx, mask),
                                              ggml_webgpu_tensor_binding_size(ctx, mask)),
            ggml_webgpu_make_bind_group_entry(1, blk_buf, scratch_offset, blk_size_bytes),
        };
        scratch_offset = ROUNDUP_POW2(scratch_offset + blk_size_bytes, align_bytes);
    }

    std::vector<uint32_t> split_params = params;
    if (has_mask) {
        split_params.push_back(0u);                     // blk_base
        split_params.push_back(blk_nblk0);              // blk_nblk0
        split_params.push_back(blk_nblk1);              // blk_nblk1
    }
    split_params.push_back(0u);                         // tmp_data_base
    split_params.push_back((uint32_t) tmp_stats_base);  // tmp_stats_base
    split_params.push_back(nwg);                        // nwg

    std::vector<wgpu::BindGroupEntry> split_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_offset(ctx, Q),
                                          ggml_webgpu_tensor_binding_size(ctx, Q)),
    };
    if (kv_overlap) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), kv_bind_offset, kv_bind_size));
    } else {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_offset(ctx, K),
                                                                  ggml_webgpu_tensor_binding_size(ctx, K)));
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_offset(ctx, V),
                                                                  ggml_webgpu_tensor_binding_size(ctx, V)));
    }
    uint32_t split_binding_index = kv_overlap ? 2u : 3u;
    if (has_mask) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(mask),
                                                                  ggml_webgpu_tensor_align_offset(ctx, mask),
                                                                  ggml_webgpu_tensor_binding_size(ctx, mask)));
    }
    if (has_sinks) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_offset(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_size(ctx, sinks)));
    }
    if (has_mask) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(split_binding_index++, blk_buf, blk_entries[1].offset, blk_size_bytes));
    }
    split_entries.push_back(
        ggml_webgpu_make_bind_group_entry(split_binding_index++, tmp_buf, tmp_bind_offset, tmp_bind_size));
    split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_offset(ctx, dst),
                                                              ggml_webgpu_tensor_binding_size(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::BindGroupEntry> reduce_entries;
    if (use_vec_reduce) {
        const uint32_t reduce_wg_size = std::max(
            32u, std::min<uint32_t>(nwg * 32u, ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = shader_lib_ctx;
        reduce_shader_ctx.max_wg_size                    = reduce_wg_size;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_base,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_group_entry(0, tmp_buf, tmp_bind_offset, tmp_size_bytes),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                              ggml_webgpu_tensor_binding_size(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    const uint64_t split_wg_total = (uint64_t) wg_x * nwg;
    GGML_ASSERT(split_wg_total <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    if (has_mask) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_count }
        });
    }
    dispatches.push_back({
        pipeline, std::move(split_params), std::move(split_entries), { (uint32_t) split_wg_total, 1u }
    });
    if (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

static webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    const bool inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    if (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        switch (unary_op) {
            case GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    float alpha_n = ggml_get_op_params_f32(dst, 1);
                    float alpha_p = ggml_get_op_params_f32(dst, 2);
                    float beta    = ggml_get_op_params_f32(dst, 3);
                    float eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    break;
                }
            default:
                break;
        }
    } else if (dst->op == GGML_OP_CLAMP) {
        float clamp_min = ggml_get_op_params_f32(dst, 0);
        float clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } else if (dst->op == GGML_OP_FILL) {
        float fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, effective_src),
    };
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_binary_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_binary_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_binary_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
    size_t src1_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src1);

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    if (decisions->src_overlap) {
        const ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_size   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        offset_src0,
        offset_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;

    if (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } else {
        entries.push_back(ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_size(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_size(ctx, src1)));
        if (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
        }
    }

    uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
        dim,
        (uint32_t) src0->ne[dim]
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    uint32_t        wg_x      = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_size(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->ne[0]),
                                     (uint32_t) (src0->ne[1]),
                                     (uint32_t) (src0->ne[2]),
                                     (uint32_t) (src0->ne[3]),
                                     (uint32_t) (dst->ne[0]),
                                     (uint32_t) (dst->ne[1]),
                                     (uint32_t) (dst->ne[2]) };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    uint32_t        wg_x      = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    if (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } else if (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } else {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t offset_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_size(rn_src->type));
    uint32_t offset_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_size(mul_src->type));
    size_t merged_offset = 0;
    size_t merged_size   = 0;

    std::vector<uint32_t> params = {
        offset_rn_src,
        offset_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_size(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_size(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_size(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_size(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_size(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_size(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) mul_src->ne[0],
        (uint32_t) mul_src->ne[1],
        (uint32_t) mul_src->ne[2],
        (uint32_t) mul_src->ne[3],
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(rn_dst, 0))  // epsilon, treated as f32 in the shader
    };

    std::vector<wgpu::BindGroupEntry> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    if (decisions->src_overlap) {
        const ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_offset  = merged_range.offset;
        merged_size    = merged_range.size;
        offset_rn_src  = ggml_webgpu_tensor_merged_element_offset(rn_src, merged_range);
        offset_mul_src = ggml_webgpu_tensor_merged_element_offset(mul_src, merged_range);
        params[0]      = offset_rn_src;
        params[1]      = offset_mul_src;
    }

    if (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
    } else if (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(rn_src), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } else {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, ggml_nrows(dst));
}

static webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_row_norm_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };
    if (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, ggml_nrows(src));
}

static webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const bool inplace         = decisions->inplace;
    const int  has_freq_factor = (src2 != nullptr);

    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    memcpy(&freq_base, (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, sizeof(float));

    int sections[4];
    memcpy(sections, (int32_t *) dst->op_params + 11, 4 * sizeof(int));

    float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        src2 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(src0) / 2,
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) n_dims,
        (uint32_t) mode,
        ggml_webgpu_u32_from_f32(theta_scale),
        ggml_webgpu_u32_from_f32(attn_factor),
        ggml_webgpu_u32_from_f32(freq_scale),
        ggml_webgpu_u32_from_f32(ext_factor),
        ggml_webgpu_u32_from_f32(corr_dims[0]),
        ggml_webgpu_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3]
    };

    std::vector<wgpu::BindGroupEntry> entries     = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    if (has_freq_factor) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
    }
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));
    }

    uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const int split = (src1 != nullptr);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],                // swapped
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 2)),  // alpha, for swiglu_oai
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 3)),  // limit, for swiglu_oai
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
    };
    uint32_t dst_binding = 1;
    if (split) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));

    uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };

    if (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const bool inplace     = decisions->inplace;
    const int  has_mask    = (src1 != nullptr);
    const int  has_sink    = (src2 != nullptr);
    float      max_bias    = ggml_get_op_params_f32(dst, 1);
    float      n_head_log2 = float(1u << (uint32_t) floor(log2(src0->ne[2])));
    float      m0          = powf(2.0f, -(max_bias) / n_head_log2);
    float      m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        has_mask ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_mask ? (uint32_t) src1->ne[2] : 0,
        has_mask ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_head_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::BindGroupEntry> entries     = { ggml_webgpu_make_bind_group_entry(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
        ggml_webgpu_tensor_binding_size(ctx, src0)) };
    uint32_t                          binding_num = 1;
    if (has_mask) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_num, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_offset(ctx, src1),
                                                            ggml_webgpu_tensor_binding_size(ctx, src1)));
        binding_num++;
    }
    if (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num, src2));
        binding_num++;
    }
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num, dst));
    }

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, ggml_nrows(dst));
}

static webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool is_top_k = dst->op == GGML_OP_TOP_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    auto * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_merge_pipeline = ctx->shader_lib->get_argsort_merge_pipeline(shader_lib_ctx);

    const uint32_t src_ne0 = (uint32_t) src->ne[0];
    const uint32_t nrows   = (uint32_t) ggml_nrows(src);
    const uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_size);
    const uint32_t block_size =
        is_top_k ? std::min(argsort_decisions->wg_size, (uint32_t) dst->ne[0]) : argsort_decisions->wg_size;
    uint32_t out_ne0 = src_ne0;
    if (is_top_k) {
        if (npr > 1) {
            const uint32_t last_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_size;
            out_ne0                  = (npr - 1) * block_size + std::min(last_tile, block_size);
        } else {
            out_ne0 = block_size;
        }
    }

    uint32_t merge_len    = block_size;
    uint32_t merge_passes = 0;
    while (merge_len < out_ne0) {
        merge_len <<= 1;
        merge_passes++;
    }

    const bool start_in_tmp = (merge_passes % 2) == 1;

    const size_t dst_offset = ggml_webgpu_tensor_offset(dst);
    const size_t idx_nbytes = out_ne0 * ggml_nrows(dst) * sizeof(int32_t);
    const size_t tmp_offset =
        ROUNDUP_POW2(dst_offset + idx_nbytes, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    const size_t tmp_binding_size = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    const size_t dst_binding_size =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    const uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type));
    const uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type));
    const uint32_t offset_tmp  = 0;
    const uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_size(src->type));
    const uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_size(src->type));
    const uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_size(src->type));
    const uint32_t stride_idx1 = out_ne0;
    const uint32_t stride_idx2 = out_ne0 * (uint32_t) dst->ne[1];
    const uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    const uint32_t init_offset       = start_in_tmp ? offset_tmp : offset_dst;
    const size_t   init_align_offset = start_in_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
    const size_t   init_binding_size = start_in_tmp ? tmp_binding_size : dst_binding_size;

    std::vector<uint32_t> init_params = {
        offset_src,  init_offset, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], out_ne0,
        block_size,  npr,         nrows
    };

    const uint32_t                    total_wg_init = npr * nrows;
    const uint32_t                    max_wg    = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    const uint32_t                    wg_x_init = std::min(total_wg_init, max_wg);
    const uint32_t                    wg_y_init = CEIL_DIV(total_wg_init, wg_x_init);
    std::vector<wgpu::BindGroupEntry> init_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), init_align_offset, init_binding_size)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    if (merge_passes == 0) {
        return ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    bool     in_is_tmp = start_in_tmp;
    uint32_t len       = block_size;
    while (len < out_ne0) {
        const uint32_t nm = CEIL_DIV(out_ne0, 2 * len);

        const bool     out_is_tmp  = !in_is_tmp;
        const uint32_t offset_in   = in_is_tmp ? offset_tmp : offset_dst;
        const uint32_t offset_out  = out_is_tmp ? offset_tmp : offset_dst;
        const size_t   align_in    = in_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        const size_t   align_out   = out_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        const size_t   size_in     = in_is_tmp ? tmp_binding_size : dst_binding_size;
        const size_t   size_out    = out_is_tmp ? tmp_binding_size : dst_binding_size;
        const uint32_t top_k_out   = (is_top_k && nm == 1) ? (uint32_t) dst->ne[0] : out_ne0;
        const uint32_t stride_out1 = top_k_out;
        const uint32_t stride_out2 = top_k_out * (uint32_t) dst->ne[1];
        const uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> merge_params = { offset_src,
                                               offset_in,
                                               offset_out,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               out_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               top_k_out,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::BindGroupEntry> merge_entries = {
            ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), align_in, size_in),
            ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), align_out, size_out)
        };

        const uint32_t total_wg_merge = nm * nrows;
        const uint32_t wg_x_merge     = std::min(total_wg_merge, max_wg);
        const uint32_t wg_y_merge     = CEIL_DIV(total_wg_merge, wg_x_merge);
        dispatches.push_back({
            argsort_merge_pipeline, std::move(merge_params), std::move(merge_entries), { wg_x_merge, wg_y_merge }
        });

        len <<= 1;
        in_is_tmp = !in_is_tmp;
    }

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

static webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nrows(dst);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_sum_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool                  total_sum = dst->op == GGML_OP_SUM;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     total_sum ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     total_sum ? 1 : (uint32_t) src->ne[1],
                                     total_sum ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_sum_rows_pipeline(shader_lib_ctx);

    uint32_t wg_x = total_sum ? 1 : ggml_nrows(dst);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static bool ggml_webgpu_can_fuse_rms_norm_mul(const struct ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        return false;
    }

    // additional constraints specific to this fusion
    const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
    const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    if (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return false;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }
    // rms_norm shader assumes contiguous rows
    if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    return true;
}

static webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    const uint32_t        mode_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

                                     (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),

                                     (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2],
                                     (uint32_t) src->ne[3],

                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],

                                     mode_flags };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    uint32_t        total_wg  = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    uint32_t        wg_x = std::min(ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, total_wg);
    uint32_t        wg_y = CEIL_DIV(total_wg, wg_x);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
static std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           int            node_idx,
                                                           int &          num_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  node  = nodes[node_idx];

    if (ggml_is_empty(node)) {
        return std::nullopt;
    }
    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << node << ", " << ggml_op_name(node->op) << ")");

    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    ggml_tensor * src2 = node->src[2];

    switch (node->op) {
            // no-ops
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            return std::nullopt;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return ggml_webgpu_cpy(ctx, src0, node);
        case GGML_OP_SET:
            return ggml_webgpu_set(ctx, src0, src1, node);
        case GGML_OP_SET_ROWS:
            return ggml_webgpu_set_rows(ctx, src0, src1, node);
        case GGML_OP_GET_ROWS:
            return ggml_webgpu_get_rows(ctx, src0, src1, node);
        case GGML_OP_MUL_MAT:
            return ggml_webgpu_mul_mat(ctx, src0, src1, node);
        case GGML_OP_MUL_MAT_ID:
            return ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, node);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_webgpu_flash_attn(ctx, src0, src1, src2, node->src[3], node->src[4], node);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return ggml_webgpu_binary_op(ctx, src0, src1, node);
        case GGML_OP_CONCAT:
            return ggml_webgpu_concat(ctx, src0, src1, node);
        case GGML_OP_REPEAT:
            return ggml_webgpu_repeat(ctx, src0, node);
        case GGML_OP_RMS_NORM:
            if (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, node_idx)) {
                num_encoded_ops        = 2;
                ggml_tensor * mul_node = nodes[node_idx + 1];
                return ggml_webgpu_rms_norm_mul(ctx, src0, node, mul_node->src[0], mul_node->src[1], mul_node);
            } else {
                return ggml_webgpu_row_norm(ctx, src0, node);
            }
        case GGML_OP_L2_NORM:
            return ggml_webgpu_row_norm(ctx, src0, node);
        case GGML_OP_ROPE:
            return ggml_webgpu_rope(ctx, src0, src1, src2, node);
        case GGML_OP_GLU:
            return ggml_webgpu_glu(ctx, src0, src1, node);
        case GGML_OP_SCALE:
            return ggml_webgpu_scale(ctx, src0, node);
        case GGML_OP_SOFT_MAX:
            return ggml_webgpu_soft_max(ctx, src0, src1, src2, node);
        case GGML_OP_UNARY:
        case GGML_OP_CLAMP:
        case GGML_OP_FILL:
        case GGML_OP_LOG:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_DIAG:
        case GGML_OP_TRI:
            return ggml_webgpu_unary_op(ctx, src0, node);
        case GGML_OP_SOLVE_TRI:
            return ggml_webgpu_solve_tri(ctx, src0, src1, node);
        case GGML_OP_SSM_CONV:
            return ggml_webgpu_ssm_conv(ctx, src0, src1, node);
        case GGML_OP_SSM_SCAN:
            return ggml_webgpu_ssm_scan(ctx, src0, src1, src2, node->src[3], node->src[4], node->src[5], node->src[6],
                                        node);
        case GGML_OP_GATED_DELTA_NET:
            return ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, node->src[3], node->src[4], node->src[5], node);
        case GGML_OP_PAD:
            return ggml_webgpu_pad(ctx, src0, node);
        case GGML_OP_ARGMAX:
            return ggml_webgpu_argmax(ctx, src0, node);
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            // we reuse the same argsort implementation for top_k
            return ggml_webgpu_argsort(ctx, src0, node);
        case GGML_OP_CUMSUM:
            return ggml_webgpu_cumsum(ctx, src0, node);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
            return ggml_webgpu_sum_rows(ctx, src0, node);
        case GGML_OP_CONV_2D:
            return ggml_webgpu_conv_2d(ctx, src0, src1, node);
        case GGML_OP_IM2COL:
            return ggml_webgpu_im2col(ctx, src0, src1, node);
        case GGML_OP_UPSCALE:
            return ggml_webgpu_upscale(ctx, src0, node);
        default:
            return std::nullopt;
    }
}

#ifdef GGML_WEBGPU_GPU_PROFILE
static void ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        const std::vector<std::string> & pipeline_names,
                                                        uint32_t &                       num_inflight_batches) {
    if (pipeline_names.empty()) {
        return;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.ResolveQuerySet(ctx->profile_timestamp_query_set, 0, ctx->profile_timestamp_query_count,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.CopyBufferToBuffer(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_count * sizeof(uint64_t));

    wgpu::CommandBuffer profile_commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, num_inflight_batches);

    const size_t mapped_size = ctx->profile_timestamp_query_count * sizeof(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_count == 2 * pipeline_names.size());

    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::MapMode::Read, 0,
                                   mapped_size);
    const uint64_t * ts_data = (const uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_size);

    for (size_t i = 0; i < pipeline_names.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        const double elapsed_ms = double(ts_data[2 * i + 1] - ts_data[2 * i]) * 1e-6;
        ctx->shader_gpu_time_ms[pipeline_names[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#endif

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
static void ggml_backend_webgpu_check_set_rows(webgpu_context & ctx, uint32_t & num_inflight_batches) {
#ifdef GGML_WEBGPU_CHECK_SET_ROWS
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->set_rows_dev_error_buf, 0, ctx->set_rows_host_error_buf, 0,
                               ctx->set_rows_host_error_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, commands, num_inflight_batches);
    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->set_rows_host_error_buf, wgpu::MapMode::Read, 0,
                                   ctx->set_rows_host_error_buf.GetSize());
    const uint32_t * error_data = (const uint32_t *) ctx->set_rows_host_error_buf.GetConstMappedRange();
    if (*error_data) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->set_rows_host_error_buf.Unmap();
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(num_inflight_batches);
#endif
}

static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_TOTAL_START(graph_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t num_batched_kernels  = 0;
    uint32_t num_inflight_batches = 0;
    bool     contains_set_rows    = false;
    bool     batch_compute_passes = true;
    int      num_encoded_ops      = 1;
    int      node_idx             = 0;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_count = 0;
    batch_compute_passes               = false;
    std::vector<std::string> profile_pipeline_names;
#endif

    ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    if (batch_compute_passes) {
        ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
    }

    while (node_idx < cgraph->n_nodes) {
        if (cgraph->nodes[node_idx]->op == GGML_OP_SET_ROWS) {
            contains_set_rows = true;
        }
        if (auto cmd = ggml_webgpu_encode(ctx, cgraph, node_idx, num_encoded_ops)) {
            commands.push_back(*cmd);
            num_batched_kernels += cmd.value().num_kernels;
#ifdef GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_names.insert(profile_pipeline_names.end(), cmd->pipeline_names.begin(),
                                          cmd->pipeline_names.end());
#endif
        }

        if (num_batched_kernels >= ctx->global_ctx->command_submit_batch_size) {
            if (ctx->active_compute_pass) {
                ctx->active_compute_pass.End();
            }
            num_batched_kernels                = 0;
            wgpu::CommandBuffer batch_commands = ctx->active_command_encoder.Finish();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, num_inflight_batches);

            // reset state for next batch
            ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            if (batch_compute_passes) {
                ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
        }

        node_idx += num_encoded_ops;
        num_encoded_ops = 1;
    }

    if (ctx->active_compute_pass) {
        ctx->active_compute_pass.End();
        ctx->active_compute_pass = nullptr;
    }

    if (num_batched_kernels > 0) {
        wgpu::CommandBuffer batch_commands = ctx->active_command_encoder.Finish();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, num_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->active_command_encoder = nullptr;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, num_inflight_batches);
#endif

    if (contains_set_rows) {
        ggml_backend_webgpu_check_set_rows(ctx, num_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);
    return GGML_STATUS_SUCCESS;
}

struct ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    bool                  recorded = false;
};

static ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    auto * event_ctx      = new ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    auto * event   = new ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    return event;
}

static void ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    delete static_cast<ggml_backend_webgpu_event_context *>(event->context);
    delete event;
}

static void ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    if (!event_ctx->recorded) {
        return;
    }
    wgpu::WaitStatus status =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    if (status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = false;
}

static void ggml_backend_webgpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {});
    event_ctx->recorded = true;
}

static void ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    ggml_backend_webgpu_device_event_synchronize(nullptr, event);
}

static void ggml_backend_webgpu_set_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 const void *   data,
                                                 size_t         offset,
                                                 size_t         size) {
    GGML_UNUSED(backend);
    auto * buf_ctx      = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // Write aligned portion
    buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);

    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        for (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((const uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
                                          total_offset + (size - remaining_size), remaining_size);
    }
}

static void ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_queue(backend_ctx->webgpu_ctx->global_ctx);
}

static ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_set_tensor_async,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_webgpu_graph_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_record,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ NULL,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

static void ggml_backend_webgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_webgpu_buffer_context * ctx = static_cast<ggml_backend_webgpu_buffer_context *>(buffer->context);
    if (ctx != nullptr && ctx->buffer != nullptr) {
        ctx->buffer.Destroy();
        delete ctx;
    }
}

// Returns the "fake" base pointer.
static void * ggml_backend_webgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return webgpu_ptr_base;
}

static void ggml_backend_webgpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                offset,
                                                     size_t                size) {
    if (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        return;
    }

    WEBGPU_CPU_PROFILE_TOTAL_START(memset_tensor);

    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << offset << ", " << size << ")");

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32, total_offset, size);
    WEBGPU_CPU_PROFILE_TOTAL_END(memset_tensor, buf_ctx->global_ctx);
}

static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *         tensor,
                                                  const void *          data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_TOTAL_START(set_tensor);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                              << ", " << offset << ", " << size << ")");

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);

    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        for (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((const uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
                                          total_offset + (size - remaining_size), remaining_size);
    }
    WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor, buf_ctx->global_ctx);
}

static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                  const ggml_tensor *   tensor,
                                                  void *                data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_TOTAL_START(get_tensor);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                              << ", " << offset << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    size_t final_size = size;
    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_size = size + (4 - (size % 4));
    }

    std::lock_guard<std::recursive_mutex> lock(buf_ctx->global_ctx->mutex);

    if (buf_ctx->global_ctx->get_tensor_staging_buf == nullptr ||
        buf_ctx->global_ctx->get_tensor_staging_buf.GetSize() < final_size) {
        // Create a new staging buffer if it doesn't exist or is too small
        if (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_buffer(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_size,
                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(buf_ctx->buffer, total_offset, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_size);
    wgpu::CommandBuffer commands = encoder.Finish();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_buffer(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::MapMode::Read, 0, final_size);
    // Must specify size here since the staging buffer might be larger than the tensor size
    const void * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_size);

    // Copy the data from the mapped range to the output buffer
    std::memcpy(data, mapped_range, size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor, buf_ctx->global_ctx);
}

static void ggml_backend_webgpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << buffer << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_TOTAL_START(clear);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, value, 0, buffer->size);
    WEBGPU_CPU_PROFILE_TOTAL_END(clear, buf_ctx->global_ctx);
}

static ggml_backend_buffer_i ggml_backend_webgpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_webgpu_buffer_get_base,
    /* .init_tensor     = */ NULL,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_buffer_clear,
    /* .reset           = */ NULL,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

static const char * ggml_backend_webgpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->device_name.c_str();
}

static ggml_backend_buffer_t ggml_backend_webgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                          size_t                     size) {
    static std::atomic<int> buffer_count;
    int                     buffer_id = buffer_count++;
    std::string             buf_name  = "tensor_buf" + std::to_string(buffer_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << buffer_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::Buffer                         buf;
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_buffer_context * buf_ctx =
        new ggml_backend_webgpu_buffer_context(buf, buf_name, ctx->webgpu_global_ctx);

    return ggml_backend_buffer_init(buft, ggml_backend_webgpu_buffer_interface, buf_ctx, size);
}

static size_t ggml_backend_webgpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return dev_ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
static size_t ggml_backend_webgpu_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return dev_ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize;
}

static size_t ggml_backend_webgpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                             const ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    switch (tensor->op) {
        case GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            break;
        case GGML_OP_TOP_K:
            {
                const ggml_tensor * src0 = tensor->src[0];
                if (src0) {
                    const size_t full = sizeof(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        full * 2 + ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                const ggml_tensor * Q     = tensor->src[0];
                const ggml_tensor * K     = tensor->src[1];
                const ggml_tensor * V     = tensor->src[2];
                const ggml_tensor * mask  = tensor->src[3];
                const ggml_tensor * sinks = tensor->src[4];
                if (Q && K && V) {
                    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
                    shader_lib_ctx.src0                           = const_cast<ggml_tensor *>(Q);
                    shader_lib_ctx.src1                           = const_cast<ggml_tensor *>(K);
                    shader_lib_ctx.src2                           = const_cast<ggml_tensor *>(V);
                    shader_lib_ctx.src3                           = const_cast<ggml_tensor *>(mask);
                    shader_lib_ctx.src4                           = const_cast<ggml_tensor *>(sinks);
                    shader_lib_ctx.dst                            = const_cast<ggml_tensor *>(tensor);
                    shader_lib_ctx.max_wg_size =
                        ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
                    shader_lib_ctx.wg_mem_limit_bytes =
                        ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
                    shader_lib_ctx.supports_subgroups = ctx->webgpu_global_ctx->capabilities.supports_subgroups;
                    shader_lib_ctx.supports_subgroup_matrix =
                        ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix;
                    shader_lib_ctx.sg_mat_m          = ctx->webgpu_global_ctx->capabilities.sg_mat_m;
                    shader_lib_ctx.sg_mat_n          = ctx->webgpu_global_ctx->capabilities.sg_mat_n;
                    shader_lib_ctx.sg_mat_k          = ctx->webgpu_global_ctx->capabilities.sg_mat_k;
                    shader_lib_ctx.max_subgroup_size = ctx->webgpu_global_ctx->capabilities.max_subgroup_size;

                    const ggml_webgpu_flash_attn_decisions decisions = ggml_webgpu_flash_attn_get_decisions(
                        shader_lib_ctx, ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);

                    if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
                        const uint32_t kv_tile = decisions.kv_tile;

                        const uint32_t vec_nwg_cap = std::max(
                            1u, std::min<uint32_t>(32u, ctx->webgpu_global_ctx->capabilities.max_subgroup_size));
                        uint32_t       nwg     = 1u;
                        const uint64_t kv_span = (uint64_t) std::max(1u, kv_tile);
                        while ((2u * nwg * kv_span) < (uint64_t) K->ne[1] && nwg < vec_nwg_cap) {
                            nwg <<= 1;
                        }
                        nwg = std::min(nwg, vec_nwg_cap);

                        const size_t align =
                            ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
                        const uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                        if (nwg > 1u) {
                            const uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                            const uint64_t tmp_stats_elems = nrows * 2u * nwg;
                            const size_t   tmp_size_bytes  = ROUNDUP_POW2(
                                (tmp_data_elems + tmp_stats_elems) * sizeof(float), WEBGPU_STORAGE_BUF_BINDING_MULT);
                            res += tmp_size_bytes + align;
                        } else {
                            res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                        }
                        if (mask != nullptr) {
                            const uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                            const uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                            const uint32_t stride_mask3    = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
                            const uint32_t blk_batch_count = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                            const uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_count;
                            const size_t   blk_size_bytes =
                                ROUNDUP_POW2(blk_elems * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                            res += blk_size_bytes + align;
                        }
                        res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                    }
                }
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            {
                const ggml_tensor * src0 = tensor->src[0];
                const ggml_tensor * src1 = tensor->src[1];
                if (src0 && src1) {
                    const size_t gathered_size = sizeof(uint32_t) * tensor->src[0]->ne[2] * tensor->src[1]->ne[2];
                    const size_t gathered_count_ids_size = sizeof(uint32_t) * tensor->src[0]->ne[2];
                    res                                  = ROUNDUP_POW2(
                        res + gathered_size * 2 + gathered_count_ids_size +
                            ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment * 3,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            break;
        default:
            break;
    }
    return res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

static const char * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    return ctx->device_name.c_str();
}

static const char * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    return ctx->device_desc.c_str();
}

static void ggml_backend_webgpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_buffer_size = ctx->webgpu_global_ctx->capabilities.limits.maxBufferSize;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#if UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_size = static_cast<uint64_t>(UINTPTR_MAX);
    if (max_buffer_size > max_ptr_size) {
        max_buffer_size = max_ptr_size;
    }
#endif
    *free  = static_cast<size_t>(max_buffer_size);
    *total = static_cast<size_t>(max_buffer_size);
}

static enum ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_guid_t ggml_backend_webgpu_guid(void) {
    static ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    return &guid;
}

static void ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_threads = WEBGPU_MAX_WG_SIZE * ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_bytes_per_thread =
        CEIL_DIV(ctx->capabilities.limits.maxStorageBufferBindingSize, max_threads);
    std::vector<wgpu::ConstantEntry> constants(2);
    constants[0].key     = "wg_size";
    constants[0].value   = WEBGPU_MAX_WG_SIZE;
    constants[1].key     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_bytes_per_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

static bool create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    wgpu::RequestAdapterOptions options = {};

#ifndef __EMSCRIPTEN__
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    const char * const          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabledToggleCount = 2;
    options.nextInChain                   = &adapterTogglesDesc;
#endif

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->instance.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [&ctx](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char * message) {
                if (status != wgpu::RequestAdapterStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
                    return;
                }
                ctx->webgpu_global_ctx->adapter = std::move(adapter);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != nullptr);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#ifndef __EMSCRIPTEN__
    wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs{};
    if (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        info.nextInChain = &subgroup_matrix_configs;
    }
#endif
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_size = ggml_backend_webgpu_get_command_submit_batch_size();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    wgpu::SupportedFeatures features;
    ctx->webgpu_global_ctx->adapter.GetFeatures(&features);
    // we require f16 support
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ShaderF16));
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);

    bool valid_subgroup_matrix_config = false;
#ifndef __EMSCRIPTEN__
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    if (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
            const wgpu::SubgroupMatrixConfig config = subgroup_matrix_configs.configs[i];
            if (config.componentType == wgpu::SubgroupMatrixComponentType::F16 &&
                config.resultComponentType == wgpu::SubgroupMatrixComponentType::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_matrix_config                  = true;
                break;
            }
        }
    }
#endif
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix = valid_subgroup_matrix_config;

    // For subgroup matrix code to be the most efficient, we would like the subgroup size to be consistent and accurate.
    // Unfortunately, that is not possible, so we use the maximum subgroup size reported by the adapter.
    ctx->webgpu_global_ctx->capabilities.max_subgroup_size = info.subgroupMaxSize;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#ifndef __EMSCRIPTEN__
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    if (ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
        required_features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
    }
#endif

    if (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#endif

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.data();
    dev_desc.requiredFeatureCount = required_features.size();
    dev_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            if (reason == wgpu::DeviceLostReason::Destroyed) {
                return;
            }
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason),
                           std::string(message).c_str());
        });
    dev_desc.SetUncapturedErrorCallback(
        [](const wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<int>(reason),
                       std::string(message).c_str());
        });

#ifndef __EMSCRIPTEN__
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    const char * const          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    const char * const          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabledToggleCount  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabledToggleCount = 1;

    dev_desc.nextInChain = &deviceTogglesDesc;
#endif

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::CallbackMode::AllowSpontaneous,
            [ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                if (status != wgpu::RequestDeviceStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(message).c_str());
                    return;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != nullptr);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_SIZE_BYTES, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.GetQueue();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
    return true;
}

static webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_SIZE_BYTES,
        webgpu_ctx->global_ctx->command_submit_batch_size + WEBGPU_NUM_PARAM_SLOT_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.minUniformBufferOffsetAlignment);
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_dev_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_host_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "set_rows_host_error_buf");

#ifdef GGML_WEBGPU_GPU_PROFILE
    ggml_webgpu_create_buffer(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
        wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "profile_timestamp_host_buf");
    wgpu::QuerySetDescriptor query_set_desc = {};
    query_set_desc.type                     = wgpu::QueryType::Timestamp;
    query_set_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_COUNT;
    webgpu_ctx->profile_timestamp_query_set = webgpu_ctx->global_ctx->device.CreateQuerySet(&query_set_desc);
#endif

#ifdef GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * sizeof(uint32_t),
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "debug_host_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * sizeof(uint32_t),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "debug_dev_buf");
#endif
    return webgpu_ctx;
}

static ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    auto * backend_ctx      = new ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    auto * backend = new ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_webgpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    static struct ggml_backend_buffer_type ggml_backend_webgpu_buffer_type = {
        /* .iface = */ {
                        /* .get_name       = */ ggml_backend_webgpu_buffer_type_get_name,
                        /* .alloc_buffer   = */ ggml_backend_webgpu_buffer_type_alloc_buffer,
                        /* .get_alignment  = */ ggml_backend_webgpu_buffer_type_get_alignment,
                        /* .get_max_size   = */ ggml_backend_webgpu_buffer_type_get_max_size,
                        /* .get_alloc_size = */ ggml_backend_webgpu_buffer_type_get_alloc_size,
                        /* .is_host        = */ NULL,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ NULL
    };

    return &ggml_backend_webgpu_buffer_type;
}

static bool ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft->iface.get_name == ggml_backend_webgpu_buffer_type_get_name;
}

static bool ggml_webgpu_supported_qtype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    if (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize)) {
        return false;
    }

    bool supports_op = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            supports_op = true;
            break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            break;
        case GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            break;
        case GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32 || src0->type == GGML_TYPE_I16);
            break;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            supports_op = ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) ||
                          (op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32);
            break;
        case GGML_OP_SET:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            break;
        case GGML_OP_SET_ROWS:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) && src0->type == GGML_TYPE_F32 &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            break;
        case GGML_OP_GET_ROWS:
            if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } else if (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            break;
        case GGML_OP_MUL_MAT:
            {
                switch (src1->type) {
                    case GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        break;
                    case GGML_TYPE_F32:
                        switch (src0->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                            case GGML_TYPE_Q1_0:
                            case GGML_TYPE_Q4_0:
                            case GGML_TYPE_Q4_1:
                            case GGML_TYPE_Q5_0:
                            case GGML_TYPE_Q5_1:
                            case GGML_TYPE_Q8_0:
                            case GGML_TYPE_Q2_K:
                            case GGML_TYPE_Q3_K:
                            case GGML_TYPE_Q4_K:
                            case GGML_TYPE_Q5_K:
                            case GGML_TYPE_Q6_K:
                            case GGML_TYPE_IQ2_XXS:
                            case GGML_TYPE_IQ2_XS:
                            case GGML_TYPE_IQ2_S:
                            case GGML_TYPE_IQ3_XXS:
                            case GGML_TYPE_IQ3_S:
                            case GGML_TYPE_IQ1_S:
                            case GGML_TYPE_IQ1_M:
                            case GGML_TYPE_IQ4_NL:
                            case GGML_TYPE_IQ4_XS:
                                supports_op = true;
                                break;
                            default:
                                break;
                        }
                    default:
                        break;
                }
                break;
            }
        case GGML_OP_MUL_MAT_ID:
            switch (src1->type) {
                case GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    break;
                case GGML_TYPE_F32:
                    switch (src0->type) {
                        case GGML_TYPE_F32:
                        case GGML_TYPE_F16:
                        case GGML_TYPE_Q1_0:
                        case GGML_TYPE_Q4_0:
                        case GGML_TYPE_Q4_1:
                        case GGML_TYPE_Q5_0:
                        case GGML_TYPE_Q5_1:
                        case GGML_TYPE_Q8_0:
                        case GGML_TYPE_Q2_K:
                        case GGML_TYPE_Q3_K:
                        case GGML_TYPE_Q4_K:
                        case GGML_TYPE_Q5_K:
                        case GGML_TYPE_Q6_K:
                        case GGML_TYPE_IQ1_S:
                        case GGML_TYPE_IQ1_M:
                        case GGML_TYPE_IQ2_XXS:
                        case GGML_TYPE_IQ2_XS:
                        case GGML_TYPE_IQ2_S:
                        case GGML_TYPE_IQ3_XXS:
                        case GGML_TYPE_IQ3_S:
                        case GGML_TYPE_IQ4_NL:
                        case GGML_TYPE_IQ4_XS:
                            supports_op = true;
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              src2->type == src1->type && op->type == GGML_TYPE_F32;
                if (!supports_op) {
                    break;
                }
                ggml_webgpu_shader_lib_context shader_lib_ctx = {};
                shader_lib_ctx.src0                           = src0;
                shader_lib_ctx.src1                           = src1;
                shader_lib_ctx.src2                           = src2;
                shader_lib_ctx.src3                           = op->src[3];
                shader_lib_ctx.src4                           = op->src[4];
                shader_lib_ctx.dst                            = const_cast<ggml_tensor *>(op);
                shader_lib_ctx.supports_subgroups             = ctx->webgpu_global_ctx->capabilities.supports_subgroups;
                shader_lib_ctx.supports_subgroup_matrix = ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix;
                shader_lib_ctx.wg_mem_limit_bytes =
                    ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
                shader_lib_ctx.sg_mat_m          = ctx->webgpu_global_ctx->capabilities.sg_mat_m;
                shader_lib_ctx.sg_mat_n          = ctx->webgpu_global_ctx->capabilities.sg_mat_n;
                shader_lib_ctx.sg_mat_k          = ctx->webgpu_global_ctx->capabilities.sg_mat_k;
                shader_lib_ctx.max_subgroup_size = ctx->webgpu_global_ctx->capabilities.max_subgroup_size;

                const ggml_webgpu_flash_attn_decisions decisions = ggml_webgpu_flash_attn_get_decisions(
                    shader_lib_ctx, ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
                const size_t limit_bytes = ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
                const bool   has_mask    = op->src[3] != nullptr;
                if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_NONE) {
                    supports_op = false;
                    break;
                }
                if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
                    const size_t min_bytes =
                        ggml_webgpu_flash_attn_wg_mem_bytes(decisions.q_tile, decisions.kv_tile, (uint32_t) src0->ne[0],
                                                            (uint32_t) src2->ne[0], has_mask, decisions.kv_direct);
                    if (min_bytes > limit_bytes) {
                        supports_op = false;
                    }
                    break;
                }

                if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
                    const size_t min_bytes =
                        ggml_webgpu_flash_attn_wg_mem_bytes(decisions.q_tile, decisions.kv_tile, (uint32_t) src0->ne[0],
                                                            (uint32_t) src2->ne[0], has_mask, decisions.kv_direct);
                    if (min_bytes > limit_bytes) {
                        supports_op = false;
                    }
                    break;
                }

                if (!ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
                    supports_op = false;
                    break;
                }
                const size_t min_bytes =
                    ggml_webgpu_flash_attn_wg_mem_bytes(decisions.q_tile, decisions.kv_tile, (uint32_t) src0->ne[0],
                                                        (uint32_t) src2->ne[0], has_mask, decisions.kv_direct);
                if (min_bytes > limit_bytes) {
                    supports_op = false;
                }
                break;
            }
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    break;
                default:
                    break;
            }
            break;
        case GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            break;
        case GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            break;
        case GGML_OP_UNARY:
            {
                const ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                switch (UNARY_OP) {
                    case GGML_UNARY_OP_ABS:
                    case GGML_UNARY_OP_SGN:
                    case GGML_UNARY_OP_NEG:
                    case GGML_UNARY_OP_STEP:
                    case GGML_UNARY_OP_TANH:
                    case GGML_UNARY_OP_ELU:
                    case GGML_UNARY_OP_RELU:
                    case GGML_UNARY_OP_SIGMOID:
                    case GGML_UNARY_OP_GELU:
                    case GGML_UNARY_OP_GELU_QUICK:
                    case GGML_UNARY_OP_SILU:
                    case GGML_UNARY_OP_HARDSWISH:
                    case GGML_UNARY_OP_HARDSIGMOID:
                    case GGML_UNARY_OP_EXP:
                    case GGML_UNARY_OP_GELU_ERF:
                    case GGML_UNARY_OP_SOFTPLUS:
                    case GGML_UNARY_OP_EXPM1:
                    case GGML_UNARY_OP_FLOOR:
                    case GGML_UNARY_OP_CEIL:
                    case GGML_UNARY_OP_ROUND:
                    case GGML_UNARY_OP_TRUNC:
                    case GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        break;
                    default:
                        break;
                }
            }
            break;
        case GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            break;
        case GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            break;
        case GGML_OP_IM2COL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            break;
        case GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            break;
        case GGML_OP_SSM_SCAN:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            break;
        case GGML_OP_GATED_DELTA_NET:
            {
                const uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            }
            break;
        case GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_FILL:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            break;
        case GGML_OP_TOP_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            break;
        case GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            break;
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_rows(src0);
            break;
        case GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            break;
        default:
            break;
    }
    if (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src2 != nullptr &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize)) {
        supports_op = false;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    if (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } else {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    return supports_op;
}

static struct ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_memory,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

static const char * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    return ctx->name;
}

static size_t ggml_backend_webgpu_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    return ctx->device_count;
}

// Only one device is supported for now
static ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_TOTAL_START(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    static ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_TOTAL_END(reg_get_device, reg_ctx->webgpu_global_ctx);
    return &device;
}

static const struct ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    static ggml_backend_webgpu_reg_context * ctx = new ggml_backend_webgpu_reg_context();

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_count = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    if (ctx->webgpu_global_ctx != nullptr && ctx->webgpu_global_ctx->instance != nullptr) {
        return &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.data();
    instance_descriptor.requiredFeatureCount                 = instance_features.size();

#ifndef __EMSCRIPTEN__
    const char * const          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabledToggleCount = 1;
    instance_descriptor.nextInChain        = &instanceTogglesDesc;
#endif

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(new webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    if (ctx->webgpu_global_ctx->instance != nullptr) {
        wgpu::RequestAdapterOptions options = {};

        // probe for adapter support
        ctx->webgpu_global_ctx->instance.WaitAny(
            ctx->webgpu_global_ctx->instance.RequestAdapter(
                &options, wgpu::CallbackMode::AllowSpontaneous,
                [&adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter _adapter, const char * message) {
                    if (status != wgpu::RequestAdapterStatus::Success) {
                        GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
                        return;
                    }
                    adapter = std::move(_adapter);
                }),
            UINT64_MAX);
    }

    if (adapter != nullptr) {
        ctx->device_count = 1;
    }

    return &reg;
}

ggml_backend_t ggml_backend_webgpu_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_webgpu_reg(), 0);

    return ggml_backend_webgpu_backend_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
