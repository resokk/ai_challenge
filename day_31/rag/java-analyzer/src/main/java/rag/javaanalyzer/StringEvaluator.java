package rag.javaanalyzer;

import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.CallableDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.CharLiteralExpr;
import com.github.javaparser.ast.expr.ConditionalExpr;
import com.github.javaparser.ast.expr.EnclosedExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.FieldAccessExpr;
import com.github.javaparser.ast.expr.IntegerLiteralExpr;
import com.github.javaparser.ast.expr.LongLiteralExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.expr.TextBlockLiteralExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.DoStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.ForEachStmt;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.resolution.declarations.ResolvedValueDeclaration;
import com.github.javaparser.symbolsolver.javaparsermodel.declarations.JavaParserFieldDeclaration;
import com.github.javaparser.symbolsolver.javaparsermodel.declarations.JavaParserParameterDeclaration;
import com.github.javaparser.symbolsolver.javaparsermodel.declarations.JavaParserVariableDeclaration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Bounded static evaluation of Java string expressions. Never executes analysed code: it only
 * folds literals, resolved final constants, effectively-final locals, conditionals and recognised
 * StringBuilder append chains. Everything else becomes a "hole" with the source expression and a
 * reason, so the result is a partial representation instead of a guess.
 */
final class StringEvaluator {

  record Segment(String text, String holeExpr, String holeReason) {
    static Segment text(String t) {
      return new Segment(t, null, null);
    }

    static Segment hole(String expr, String reason) {
      return new Segment(null, expr, reason);
    }

    boolean isHole() {
      return holeExpr != null;
    }
  }

  record Variant(List<Segment> segments, List<String> conditions) {
    static Variant of(Segment s) {
      return new Variant(List.of(s), List.of());
    }

    Variant concat(Variant other) {
      final var segs = new ArrayList<>(segments);
      segs.addAll(other.segments);
      final var conds = new ArrayList<>(conditions);
      conds.addAll(other.conditions);
      return new Variant(segs, conds);
    }

    Variant withCondition(String c) {
      final var conds = new ArrayList<>(conditions);
      conds.add(c);
      return new Variant(segments, conds);
    }

    /** Text with every hole rendered as an opaque token, used only for candidate matching. */
    String matchText() {
      final var sb = new StringBuilder();
      for (final var s : segments) {
        sb.append(s.isHole() ? "__HOLE__" : s.text);
      }
      return sb.toString();
    }
  }

  private final int maxDepth;
  private final int maxVariants;
  private final int maxLoopUnroll;
  final List<String> diagnostics = new ArrayList<>();

  StringEvaluator(int maxDepth, int maxVariants, int maxLoopUnroll) {
    this.maxDepth = maxDepth;
    this.maxVariants = maxVariants;
    this.maxLoopUnroll = maxLoopUnroll;
  }

  List<Variant> eval(Expression e) {
    return eval(e, 0, new HashSet<>());
  }

  private List<Variant> eval(Expression e, int depth, Set<String> visiting) {
    if (depth > maxDepth) {
      diagnostics.add("max_depth_exceeded at " + shortText(e));
      return hole(e, "max evaluation depth exceeded");
    }
    if (e instanceof StringLiteralExpr s) {
      return List.of(Variant.of(Segment.text(s.asString())));
    }
    if (e instanceof TextBlockLiteralExpr tb) {
      return List.of(Variant.of(Segment.text(tb.asString())));
    }
    if (e instanceof CharLiteralExpr c) {
      return List.of(Variant.of(Segment.text(String.valueOf(c.asChar()))));
    }
    if (e instanceof IntegerLiteralExpr || e instanceof LongLiteralExpr) {
      return List.of(Variant.of(Segment.text(e.toString().replaceAll("[lL]$", ""))));
    }
    if (e instanceof EnclosedExpr en) {
      return eval(en.getInner(), depth + 1, visiting);
    }
    if (e instanceof BinaryExpr b && b.getOperator() == BinaryExpr.Operator.PLUS) {
      return product(eval(b.getLeft(), depth + 1, visiting), eval(b.getRight(), depth + 1, visiting));
    }
    if (e instanceof ConditionalExpr c) {
      final var out = new ArrayList<Variant>();
      final String cond = c.getCondition().toString();
      for (final var v : eval(c.getThenExpr(), depth + 1, visiting)) {
        out.add(v.withCondition(cond));
      }
      for (final var v : eval(c.getElseExpr(), depth + 1, visiting)) {
        out.add(v.withCondition("!(" + cond + ")"));
      }
      return bound(out);
    }
    if (e instanceof NameExpr || e instanceof FieldAccessExpr) {
      return evalReference(e, depth, visiting);
    }
    if (e instanceof MethodCallExpr m) {
      return evalCall(m, depth, visiting);
    }
    return hole(e, "unsupported expression " + e.getClass().getSimpleName());
  }

  private List<Variant> evalReference(Expression e, int depth, Set<String> visiting) {
    final ResolvedValueDeclaration decl;
    try {
      decl = e.isNameExpr() ? e.asNameExpr().resolve() : e.asFieldAccessExpr().resolve();
    } catch (RuntimeException | StackOverflowError ex) {
      diagnostics.add("unresolved_symbol " + e + ": " + ex.getClass().getSimpleName());
      return hole(e, "symbol not resolved");
    }
    final String key = decl.getClass().getSimpleName() + ":" + decl.getName() + "@" + e;
    if (!visiting.add(key)) {
      return hole(e, "cyclic reference");
    }
    try {
      if (decl instanceof JavaParserFieldDeclaration f) {
        final VariableDeclarator vd = f.getVariableDeclarator();
        final boolean isFinal =
            vd.getParentNode()
                .filter(FieldDeclaration.class::isInstance)
                .map(p -> ((FieldDeclaration) p).isFinal())
                .orElse(false);
        // interface fields are implicitly final
        final boolean inInterface = f.declaringType().isInterface();
        if (!isFinal && !inInterface) {
          return hole(e, "non-final field " + decl.getName());
        }
        if (vd.getInitializer().isEmpty()) {
          return hole(e, "final field without initializer (assigned in constructor)");
        }
        return eval(vd.getInitializer().get(), depth + 1, visiting);
      }
      if (decl instanceof JavaParserVariableDeclaration v) {
        final VariableDeclarator vd = v.getVariableDeclarator();
        if (isStringBuilderType(vd)) {
          return hole(e, "StringBuilder used as value; only toString() is evaluated");
        }
        if (isReassigned(vd)) {
          return hole(e, "local variable reassigned");
        }
        if (vd.getInitializer().isEmpty()) {
          return hole(e, "local variable without initializer");
        }
        return eval(vd.getInitializer().get(), depth + 1, visiting);
      }
      if (decl instanceof JavaParserParameterDeclaration) {
        return List.of(Variant.of(Segment.hole(e.toString(), "parameter")));
      }
      return hole(e, "declaration not available as source (" + decl.getClass().getSimpleName() + ")");
    } catch (RuntimeException ex) {
      diagnostics.add("resolution_error " + e + ": " + ex.getClass().getSimpleName());
      return hole(e, "resolution error");
    } finally {
      visiting.remove(key);
    }
  }

  private List<Variant> evalCall(MethodCallExpr m, int depth, Set<String> visiting) {
    final String name = m.getNameAsString();
    if (name.equals("toString") && m.getArguments().isEmpty() && m.getScope().isPresent()) {
      final var scope = m.getScope().get();
      if (scope.isNameExpr()) {
        final var builder = findBuilderDeclaration(scope.asNameExpr());
        if (builder.isPresent()) {
          return evalBuilder(builder.get(), m, depth, visiting);
        }
      }
    }
    if (name.equals("valueOf")
        && m.getScope().map(s -> s.toString().equals("String")).orElse(false)
        && m.getArguments().size() == 1) {
      return eval(m.getArgument(0), depth + 1, visiting);
    }
    if (name.equals("name") && m.getArguments().isEmpty() && m.getScope().isPresent()) {
      // Enum.name() of a statically known constant
      final var scope = m.getScope().get();
      if (scope.isFieldAccessExpr()) {
        return List.of(Variant.of(Segment.text(scope.asFieldAccessExpr().getNameAsString())));
      }
    }
    return hole(m, "method call " + name + "()");
  }

  private Optional<VariableDeclarator> findBuilderDeclaration(NameExpr n) {
    try {
      final var d = n.resolve();
      if (d instanceof JavaParserVariableDeclaration v && isStringBuilderType(v.getVariableDeclarator())) {
        return Optional.of(v.getVariableDeclarator());
      }
    } catch (RuntimeException ex) {
      diagnostics.add("unresolved_builder " + n + ": " + ex.getClass().getSimpleName());
    }
    return Optional.empty();
  }

  private static boolean isStringBuilderType(VariableDeclarator vd) {
    final String t = vd.getTypeAsString();
    if (t.equals("StringBuilder") || t.equals("StringBuffer")) {
      return true;
    }
    Expression init = vd.getInitializer().orElse(null);
    while (init instanceof MethodCallExpr m && m.getNameAsString().equals("append") && m.getScope().isPresent()) {
      init = m.getScope().get();
    }
    return t.equals("var") && init instanceof ObjectCreationExpr oc && oc.getTypeAsString().matches("StringBu(ilder|ffer)");
  }

  private static boolean isReassigned(VariableDeclarator vd) {
    final var body = vd.findAncestor(CallableDeclaration.class).orElse(null);
    if (body == null) {
      return false;
    }
    final String name = vd.getNameAsString();
    return body.findAll(AssignExpr.class).stream()
        .anyMatch(a -> a.getTarget().isNameExpr() && a.getTarget().asNameExpr().getNameAsString().equals(name));
  }

  /** Evaluates the append chain of a local StringBuilder up to the toString() call. */
  private List<Variant> evalBuilder(VariableDeclarator vd, MethodCallExpr toStringCall, int depth, Set<String> visiting) {
    final var block = vd.findAncestor(BlockStmt.class).orElse(null);
    if (block == null) {
      return hole(toStringCall, "builder declared outside a block");
    }
    final String name = vd.getNameAsString();
    List<Variant> state = List.of(new Variant(List.of(), List.of()));
    // initializer may itself be a chain: new StringBuilder("a").append(b)
    Expression init = vd.getInitializer().orElse(null);
    final var initAppends = new ArrayList<Expression>();
    while (init instanceof MethodCallExpr m && m.getNameAsString().equals("append") && m.getArguments().size() == 1
        && m.getScope().isPresent()) {
      initAppends.add(0, m.getArgument(0));
      init = m.getScope().get();
    }
    if (init instanceof ObjectCreationExpr oc && oc.getArguments().size() == 1 && !oc.getArgument(0).isIntegerLiteralExpr()) {
      state = eval(oc.getArgument(0), depth + 1, visiting);
    }
    for (final var arg : initAppends) {
      state = product(state, eval(arg, depth + 1, visiting));
    }
    return walkStatements(block.getStatements(), name, toStringCall, state, depth, visiting);
  }

  private List<Variant> walkStatements(
      List<Statement> stmts, String sb, Node stop, List<Variant> state, int depth, Set<String> visiting) {
    for (final Statement s : stmts) {
      if (!s.getRange().isPresent() || !stop.getRange().isPresent()) {
        break;
      }
      final var sRange = s.getRange().get();
      final var stopBegin = stop.getRange().get().begin;
      if (sRange.begin.isAfter(stopBegin)) {
        break;
      }
      final boolean containsStop = sRange.contains(stop.getRange().get());
      state = walkStatement(s, sb, stop, containsStop, state, depth, visiting);
      if (containsStop) {
        break;
      }
    }
    return state;
  }

  private List<Variant> walkStatement(
      Statement s, String sb, Node stop, boolean containsStop, List<Variant> state, int depth, Set<String> visiting) {
    if (!mentions(s, sb)) {
      return state;
    }
    if (s instanceof ExpressionStmt es) {
      final var appends = appendArguments(es.getExpression(), sb);
      if (appends == null) {
        if (containsStop) {
          return state;
        }
        diagnostics.add("unsupported_builder_operation " + shortText(es));
        return product(state, hole(es.getExpression(), "unsupported builder operation"));
      }
      for (final var arg : appends) {
        state = product(state, eval(arg, depth + 1, visiting));
      }
      return state;
    }
    if (s instanceof BlockStmt b) {
      return walkStatements(b.getStatements(), sb, stop, state, depth, visiting);
    }
    if (s instanceof IfStmt is) {
      final String cond = is.getCondition().toString();
      final var thenState = walkStatement(is.getThenStmt(), sb, stop, containsStop, withCondition(state, cond), depth, visiting);
      final var elseBase = withCondition(state, "!(" + cond + ")");
      final var elseState =
          is.getElseStmt().map(e -> walkStatement(e, sb, stop, containsStop, elseBase, depth, visiting)).orElse(elseBase);
      final var out = new ArrayList<>(thenState);
      out.addAll(elseState);
      return bound(out);
    }
    if (s instanceof ForStmt || s instanceof ForEachStmt || s instanceof WhileStmt || s instanceof DoStmt) {
      final Statement body = ((com.github.javaparser.ast.nodeTypes.NodeWithBody<?>) s).getBody();
      final var out = new ArrayList<Variant>();
      List<Variant> iter = state;
      for (int i = 0; i <= maxLoopUnroll; i++) {
        final int k = i;
        out.addAll(iter.stream().map(v -> v.withCondition("loop@" + s.getBegin().map(p -> p.line).orElse(-1) + " iterations=" + k)).toList());
        iter = walkStatement(body, sb, stop, containsStop, iter, depth, visiting);
      }
      if (maxLoopUnroll >= 0) {
        diagnostics.add("loop_unrolled_bounded max=" + maxLoopUnroll);
      }
      return bound(out);
    }
    diagnostics.add("unsupported_builder_statement " + s.getClass().getSimpleName());
    return product(state, List.of(Variant.of(Segment.hole(shortText(s), "builder used in unsupported statement"))));
  }

  /** Returns append arguments of `sb.append(a).append(b)` or null if not a pure append chain. */
  private static List<Expression> appendArguments(Expression e, String sb) {
    if (e instanceof VariableDeclarationExpr) {
      return null;
    }
    final var args = new ArrayList<Expression>();
    Expression cur = e;
    while (cur instanceof MethodCallExpr m && m.getNameAsString().equals("append") && m.getArguments().size() == 1) {
      args.add(0, m.getArgument(0));
      if (m.getScope().isEmpty()) {
        return null;
      }
      cur = m.getScope().get();
    }
    if (cur instanceof NameExpr n && n.getNameAsString().equals(sb) && !args.isEmpty()) {
      return args;
    }
    return null;
  }

  private static boolean mentions(Node n, String name) {
    return n.findAll(NameExpr.class).stream().anyMatch(x -> x.getNameAsString().equals(name));
  }

  private List<Variant> withCondition(List<Variant> state, String cond) {
    return state.stream().map(v -> v.withCondition(cond)).toList();
  }

  private List<Variant> product(List<Variant> left, List<Variant> right) {
    final var out = new ArrayList<Variant>();
    for (final var l : left) {
      for (final var r : right) {
        out.add(l.concat(r));
      }
    }
    return bound(out);
  }

  private List<Variant> bound(List<Variant> vs) {
    if (vs.size() <= maxVariants) {
      return vs;
    }
    diagnostics.add("variant_limit_reached kept=" + maxVariants + " of=" + vs.size());
    return new ArrayList<>(vs.subList(0, maxVariants));
  }

  private static List<Variant> hole(Node e, String reason) {
    return List.of(Variant.of(Segment.hole(shortText(e), reason)));
  }

  static String shortText(Node n) {
    final String s = n.toString().replaceAll("\\s+", " ");
    return s.length() > 200 ? s.substring(0, 200) + "…" : s;
  }
}
