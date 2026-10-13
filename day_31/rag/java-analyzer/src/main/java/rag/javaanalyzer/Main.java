package rag.javaanalyzer;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.body.CallableDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.RecordDeclaration;
import com.github.javaparser.ast.body.TypeDeclaration;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NormalAnnotationExpr;
import com.github.javaparser.ast.expr.SingleMemberAnnotationExpr;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.expr.TextBlockLiteralExpr;
import com.github.javaparser.ast.nodeTypes.NodeWithJavadoc;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;

/**
 * Analyses Java sources with JavaParser + Symbol Solver and writes versioned JSONL records:
 * header, file (status per input file), query (string/annotation/builder query candidates),
 * type (model/DTO/entity classes, mapper interfaces, column-family enums).
 *
 * <p>Usage: java -jar java-analyzer.jar --repo DIR --files LIST --namespaces LIST --out FILE
 * [--max-depth N] [--max-variants N] [--max-loop N]
 */
public final class Main {

  static final int SCHEMA_VERSION = 1;
  static final String JAVAPARSER_VERSION = "3.26.4";

  // Regexes only select candidates; the structure comes from the AST and SQLGlot downstream.
  static final Pattern SQL_CANDIDATE =
      Pattern.compile(
          "(?is)^\\s*(\\(\\s*)?(SELECT\\s.+\\sFROM\\s|INSERT\\s+INTO\\s|UPDATE\\s+\\S+\\s+SET\\s|DELETE\\s+FROM\\s|MERGE\\s+INTO\\s|"
              + "CREATE\\s+(UNIQUE\\s+)?(TABLE|INDEX|VIEW)\\s|ALTER\\s+TABLE\\s|DROP\\s+(TABLE|INDEX)\\s|TRUNCATE\\s+(TABLE\\s+)?\\S|WITH\\s+\\w+\\s+AS\\s*\\()");
  static final Pattern ES_DSL_CANDIDATE =
      Pattern.compile("(?s)^\\s*\\{.*\"(query|bool|match|term|terms|range|aggs|aggregations|mappings)\"\\s*:");
  static final Set<String> QUERY_ANNOTATIONS =
      Set.of("Select", "Insert", "Update", "Delete", "Query", "NamedQuery", "NamedNativeQuery", "SqlQuery", "SqlUpdate");
  static final Pattern MODEL_TYPE = Pattern.compile(".*(DbModel|DbQuery|Entity|Dto|ColumnFamilies)$");

  private final Path repo;
  private final Set<String> mapperNamespaces;
  private final int maxDepth;
  private final int maxVariants;
  private final int maxLoop;
  private final JavaParser parser;
  private final Gson gson = new GsonBuilder().disableHtmlEscaping().create();

  Main(Path repo, Set<String> mapperNamespaces, List<Path> sourceRoots, int maxDepth, int maxVariants, int maxLoop) {
    this.repo = repo;
    this.mapperNamespaces = mapperNamespaces;
    this.maxDepth = maxDepth;
    this.maxVariants = maxVariants;
    this.maxLoop = maxLoop;
    final var solver = new CombinedTypeSolver(new ReflectionTypeSolver(true));
    for (final Path root : sourceRoots) {
      solver.add(new JavaParserTypeSolver(root, new ParserConfiguration().setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_21)));
    }
    final var config =
        new ParserConfiguration()
            .setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_21)
            .setSymbolResolver(new JavaSymbolSolver(solver));
    this.parser = new JavaParser(config);
  }

  public static void main(String[] args) throws IOException {
    final Map<String, String> opts = new HashMap<>();
    for (int i = 0; i + 1 < args.length; i += 2) {
      opts.put(args[i], args[i + 1]);
    }
    final Path repo = Path.of(require(opts, "--repo"));
    final List<String> files = Files.readAllLines(Path.of(require(opts, "--files")));
    final Set<String> namespaces =
        opts.containsKey("--namespaces")
            ? new HashSet<>(Files.readAllLines(Path.of(opts.get("--namespaces"))))
            : Set.of();
    final Path out = Path.of(require(opts, "--out"));
    final int maxDepth = Integer.parseInt(opts.getOrDefault("--max-depth", "8"));
    final int maxVariants = Integer.parseInt(opts.getOrDefault("--max-variants", "8"));
    final int maxLoop = Integer.parseInt(opts.getOrDefault("--max-loop", "1"));
    final List<Path> roots = sourceRoots(repo, opts.get("--source-roots"));

    final var main = new Main(repo, namespaces, roots, maxDepth, maxVariants, maxLoop);
    try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
      final var header = new JsonObject();
      header.addProperty("type", "header");
      header.addProperty("schema_version", SCHEMA_VERSION);
      header.addProperty("tool", "java-analyzer");
      header.addProperty("javaparser_version", JAVAPARSER_VERSION);
      header.addProperty("max_depth", maxDepth);
      header.addProperty("max_variants", maxVariants);
      header.addProperty("max_loop_unroll", maxLoop);
      header.addProperty("source_roots", roots.size());
      main.write(w, header);
      int done = 0;
      for (final String rel : files) {
        if (rel.isBlank()) {
          continue;
        }
        main.analyzeFile(w, rel);
        if (++done % 500 == 0) {
          System.err.println("java-analyzer: " + done + "/" + files.size());
        }
      }
    }
  }

  private static String require(Map<String, String> opts, String key) {
    final String v = opts.get(key);
    if (v == null) {
      throw new IllegalArgumentException("missing " + key);
    }
    return v;
  }

  private static List<Path> sourceRoots(Path repo, String listFile) throws IOException {
    final var roots = new TreeSet<Path>();
    if (listFile != null) {
      for (final String l : Files.readAllLines(Path.of(listFile))) {
        if (!l.isBlank()) {
          roots.add(repo.resolve(l));
        }
      }
    }
    return new ArrayList<>(roots);
  }

  private void write(BufferedWriter w, JsonObject o) throws IOException {
    w.write(gson.toJson(o));
    w.newLine();
  }

  void analyzeFile(BufferedWriter w, String rel) throws IOException {
    final var fileRec = new JsonObject();
    fileRec.addProperty("type", "file");
    fileRec.addProperty("path", rel);
    final CompilationUnit cu;
    try {
      final var result = parser.parse(repo.resolve(rel));
      if (!result.isSuccessful() || result.getResult().isEmpty()) {
        fileRec.addProperty("status", "parse_error");
        fileRec.addProperty("message", result.getProblems().isEmpty() ? "unknown" : result.getProblems().get(0).getMessage());
        write(w, fileRec);
        return;
      }
      cu = result.getResult().get();
    } catch (Exception | StackOverflowError e) {
      fileRec.addProperty("status", "parse_error");
      fileRec.addProperty("message", e.getClass().getSimpleName() + ": " + e.getMessage());
      write(w, fileRec);
      return;
    }
    int records = 0;
    final var diagnostics = new JsonArray();
    for (final var rec : queryRecords(cu, rel, diagnostics)) {
      write(w, rec);
      records++;
    }
    for (final TypeDeclaration<?> t : cu.findAll(TypeDeclaration.class)) {
      final String fqn = t.getFullyQualifiedName().orElse(t.getNameAsString());
      final boolean mapper = mapperNamespaces.contains(fqn);
      if (mapper || MODEL_TYPE.matcher(t.getNameAsString()).matches()) {
        write(w, typeRecord(t, fqn, rel, mapper));
        records++;
      }
    }
    fileRec.addProperty("status", "parsed");
    fileRec.addProperty("records", records);
    fileRec.add("diagnostics", diagnostics);
    write(w, fileRec);
  }

  private List<JsonObject> queryRecords(CompilationUnit cu, String rel, JsonArray fileDiagnostics) {
    final var out = new ArrayList<JsonObject>();
    final Set<Node> seen = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
    // 1) annotation queries
    for (final AnnotationExpr a : cu.findAll(AnnotationExpr.class)) {
      if (!QUERY_ANNOTATIONS.contains(a.getNameAsString())) {
        continue;
      }
      Expression value = null;
      boolean nativeQuery = false;
      if (a instanceof SingleMemberAnnotationExpr s) {
        value = s.getMemberValue();
      } else if (a instanceof NormalAnnotationExpr n) {
        for (final var p : n.getPairs()) {
          if (p.getNameAsString().equals("value") || p.getNameAsString().equals("query")) {
            value = p.getValue();
          } else if (p.getNameAsString().equals("nativeQuery")) {
            nativeQuery = p.getValue().toString().equals("true");
          }
        }
      }
      if (value == null) {
        continue;
      }
      value.walk(seen::add);
      final String hint =
          switch (a.getNameAsString()) {
            case "Query", "NamedQuery" -> nativeQuery ? "sql" : "jpql";
            default -> "sql";
          };
      final var rec = queryRecord(value, rel, "annotation", "@" + a.getNameAsString(), hint, true);
      if (rec != null) {
        out.add(rec);
      }
    }
    // 2) StringBuilder.toString() calls
    for (final MethodCallExpr m : cu.findAll(MethodCallExpr.class)) {
      if (m.getNameAsString().equals("toString") && m.getArguments().isEmpty()
          && m.getScope().map(Expression::isNameExpr).orElse(false)) {
        final var rec = queryRecord(m, rel, "string_builder", sinkOf(m), null, false);
        if (rec != null) {
          out.add(rec);
        }
      }
    }
    // 3) top-level string expressions: literals, text blocks and concatenations
    for (final Expression e : cu.findAll(Expression.class)) {
      if (seen.contains(e)) {
        continue;
      }
      final boolean literal = e instanceof StringLiteralExpr || e instanceof TextBlockLiteralExpr;
      final boolean concat =
          e instanceof BinaryExpr b && b.getOperator() == BinaryExpr.Operator.PLUS
              && (!b.findAll(StringLiteralExpr.class).isEmpty() || !b.findAll(TextBlockLiteralExpr.class).isEmpty());
      if (!literal && !concat) {
        continue;
      }
      if (e.getParentNode().map(p -> p instanceof BinaryExpr pb && pb.getOperator() == BinaryExpr.Operator.PLUS).orElse(false)) {
        continue; // part of a larger concatenation
      }
      if (e.getParentNode().map(p -> p instanceof MethodCallExpr pm && pm.getNameAsString().equals("append")
          || p instanceof com.github.javaparser.ast.expr.ObjectCreationExpr oc && oc.getTypeAsString().matches("StringBu(ilder|ffer)"))
          .orElse(false)) {
        continue; // covered by the StringBuilder evaluation of the enclosing builder
      }
      final String kind = e instanceof TextBlockLiteralExpr ? "text_block" : literal ? "literal" : "concatenation";
      final var rec = queryRecord(e, rel, kind, sinkOf(e), null, false);
      if (rec != null) {
        out.add(rec);
      }
    }
    return out;
  }

  private JsonObject queryRecord(Expression e, String rel, String kind, String sink, String hint, boolean forced) {
    final var ev = new StringEvaluator(maxDepth, maxVariants, maxLoop);
    final List<StringEvaluator.Variant> variants = ev.eval(e);
    String language = hint;
    boolean matched = false;
    for (final var v : variants) {
      final String text = v.matchText();
      if (SQL_CANDIDATE.matcher(text).find()) {
        matched = true;
        if (language == null) {
          language = sink != null && sink.contains("createQuery") ? "jpql" : "sql";
        }
      } else if (ES_DSL_CANDIDATE.matcher(text).find()) {
        matched = true;
        if (language == null) {
          language = "es_dsl";
        }
      }
    }
    if (!matched && !forced) {
      return null;
    }
    final var rec = new JsonObject();
    rec.addProperty("type", "query");
    rec.addProperty("path", rel);
    rec.addProperty("kind", kind);
    rec.addProperty("language_hint", language == null ? "unknown" : language);
    rec.addProperty("sink", sink);
    addRange(rec, e);
    addContext(rec, e);
    rec.addProperty("expression", StringEvaluator.shortText(e));
    final var vs = new JsonArray();
    for (final var v : variants) {
      final var vo = new JsonObject();
      final var segs = new JsonArray();
      boolean partial = false;
      for (final var s : v.segments()) {
        final var so = new JsonObject();
        if (s.isHole()) {
          partial = true;
          so.addProperty("hole", s.holeExpr());
          so.addProperty("reason", s.holeReason());
        } else {
          so.addProperty("text", s.text());
        }
        segs.add(so);
      }
      vo.add("segments", segs);
      vo.add("conditions", gson.toJsonTree(v.conditions()));
      vo.addProperty("partial", partial);
      vs.add(vo);
    }
    rec.add("variants", vs);
    rec.add("diagnostics", gson.toJsonTree(ev.diagnostics));
    return rec;
  }

  private static String sinkOf(Node e) {
    Node cur = e;
    while (cur.getParentNode().isPresent()) {
      final Node p = cur.getParentNode().get();
      if (p instanceof MethodCallExpr m && m.getArguments().contains(cur)) {
        return m.getNameAsString();
      }
      if (p instanceof com.github.javaparser.ast.body.VariableDeclarator vd) {
        return "var:" + vd.getNameAsString();
      }
      if (p instanceof com.github.javaparser.ast.stmt.Statement || p instanceof BodyDeclaration<?>) {
        return null;
      }
      cur = p;
    }
    return null;
  }

  private static void addRange(JsonObject o, Node n) {
    n.getRange().ifPresent(r -> {
      o.addProperty("line_start", r.begin.line);
      o.addProperty("line_end", r.end.line);
    });
  }

  private static void addContext(JsonObject rec, Node e) {
    e.findAncestor(TypeDeclaration.class)
        .ifPresent(t -> rec.addProperty("class", (String) ((TypeDeclaration<?>) t).getFullyQualifiedName().orElse(t.getNameAsString())));
    final var callable = e.findAncestor(CallableDeclaration.class);
    if (callable.isPresent()) {
      final CallableDeclaration<?> c = callable.get();
      rec.addProperty("method", c.getDeclarationAsString(false, false, true));
      c.getJavadoc().ifPresent(j -> rec.addProperty("javadoc", j.toText()));
    } else {
      e.findAncestor(FieldDeclaration.class).ifPresent(f -> {
        rec.addProperty("field", f.getVariables().get(0).getNameAsString());
        f.getJavadoc().ifPresent(j -> rec.addProperty("javadoc", j.toText()));
      });
    }
    e.findAncestor(com.github.javaparser.ast.stmt.Statement.class)
        .flatMap(Node::getComment)
        .ifPresent(c -> rec.addProperty("comment", c.getContent().strip()));
  }

  private JsonObject typeRecord(TypeDeclaration<?> t, String fqn, String rel, boolean mapper) {
    final var o = new JsonObject();
    o.addProperty("type", "type");
    o.addProperty("path", rel);
    o.addProperty("fqn", fqn);
    o.addProperty("name", t.getNameAsString());
    o.addProperty("kind",
        t instanceof EnumDeclaration ? "enum"
            : t instanceof RecordDeclaration ? "record"
            : t instanceof ClassOrInterfaceDeclaration c && c.isInterface() ? "interface" : "class");
    o.addProperty("mapper_interface", mapper);
    addRange(o, t);
    javadoc(t).ifPresent(j -> o.addProperty("javadoc", j));
    t.getComment().filter(c -> !c.isJavadocComment()).ifPresent(c -> o.addProperty("comment", c.getContent().strip()));
    o.add("annotations", gson.toJsonTree(t.getAnnotations().stream().map(AnnotationExpr::getNameAsString).toList()));

    final var fields = new JsonArray();
    if (t instanceof RecordDeclaration r) {
      for (final var p : r.getParameters()) {
        final var f = new JsonObject();
        f.addProperty("name", p.getNameAsString());
        f.addProperty("java_type", p.getTypeAsString());
        p.getBegin().ifPresent(b -> f.addProperty("line", b.line));
        fields.add(f);
      }
    }
    for (final FieldDeclaration fd : t.getFields()) {
      if (fd.isStatic()) {
        continue;
      }
      for (final var v : fd.getVariables()) {
        final var f = new JsonObject();
        f.addProperty("name", v.getNameAsString());
        f.addProperty("java_type", v.getTypeAsString());
        fd.getBegin().ifPresent(b -> f.addProperty("line", b.line));
        fd.getJavadoc().ifPresent(j -> f.addProperty("javadoc", j.toText()));
        fd.getComment().filter(c -> !c.isJavadocComment()).ifPresent(c -> f.addProperty("comment", c.getContent().strip()));
        fields.add(f);
      }
    }
    o.add("fields", fields);

    if (t instanceof EnumDeclaration en) {
      final var constants = new JsonArray();
      for (final var c : en.getEntries()) {
        final var co = new JsonObject();
        co.addProperty("name", c.getNameAsString());
        co.add("arguments", gson.toJsonTree(c.getArguments().stream().map(Node::toString).toList()));
        co.addProperty("deprecated", c.getAnnotationByName("Deprecated").isPresent());
        c.getComment().ifPresent(cm -> co.addProperty("comment", cm.getContent().strip()));
        c.getBegin().ifPresent(b -> co.addProperty("line", b.line));
        constants.add(co);
      }
      o.add("enum_constants", constants);
    }

    if (mapper) {
      final var methods = new JsonArray();
      for (final MethodDeclaration m : t.getMethods()) {
        final var mo = new JsonObject();
        mo.addProperty("name", m.getNameAsString());
        mo.addProperty("return_type", m.getTypeAsString());
        mo.addProperty("signature", m.getDeclarationAsString(false, false, true));
        addRange(mo, m);
        m.getJavadoc().ifPresent(j -> mo.addProperty("javadoc", j.toText()));
        mo.add("annotations", gson.toJsonTree(m.getAnnotations().stream().map(AnnotationExpr::toString).toList()));
        final var params = new JsonArray();
        for (final var p : m.getParameters()) {
          final var po = new JsonObject();
          po.addProperty("name", p.getNameAsString());
          po.addProperty("java_type", p.getTypeAsString());
          p.getAnnotationByName("Param").ifPresent(a -> {
            if (a instanceof SingleMemberAnnotationExpr s && s.getMemberValue() instanceof StringLiteralExpr sl) {
              po.addProperty("param_name", sl.asString());
            }
          });
          params.add(po);
        }
        mo.add("params", params);
        methods.add(mo);
      }
      o.add("methods", methods);
    }
    return o;
  }

  private static java.util.Optional<String> javadoc(TypeDeclaration<?> t) {
    if (t instanceof NodeWithJavadoc<?> j) {
      return j.getJavadoc().map(d -> d.toText());
    }
    return java.util.Optional.empty();
  }
}
