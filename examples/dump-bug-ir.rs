// Debug: Build e-graph manually for i & 1 == 0 with the shared state from the outer block
use jules::compiler::ast::*;
use jules::compiler::parser::Parser;
use jules::compiler::lexer::Lexer;
use jules::optimizer::advanced_optimizer::EGraph;

fn main() {
    let src = r#"fn main() -> i32 {
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < 10 {
    if i & 1 == 0 {
      s = s + i;
    } else {
      s = s - 1;
    }
    i = i + 1;
  }
  s
}"#;

    let mut lexer = Lexer::new(src);
    let (tokens, _) = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program();

    // Build a shared e-graph and add expressions in order, like the optimizer does
    let mut eg = EGraph::new();

    // Process let mut s = 0
    let s_init: Expr = Expr::IntLit { span: jules::compiler::lexer::Span::dummy(), value: 0, ty: None };
    let s_init_cls = eg.build_expr(&s_init);
    println!("After s=0: IntLit(0) -> class {}", s_init_cls);

    // Process let mut i = 0
    let i_init: Expr = Expr::IntLit { span: jules::compiler::lexer::Span::dummy(), value: 0, ty: None };
    let i_init_cls = eg.build_expr(&i_init);
    println!("After i=0: IntLit(0) -> class {}", i_init_cls);

    // Process while condition: i < 10
    let while_cond: Expr = Expr::BinOp {
        span: jules::compiler::lexer::Span::dummy(),
        op: BinOpKind::Lt,
        lhs: Box::new(Expr::Ident { span: jules::compiler::lexer::Span::dummy(), name: "i".into() }),
        rhs: Box::new(Expr::IntLit { span: jules::compiler::lexer::Span::dummy(), value: 10, ty: None }),
    };
    // But wait - the while condition is NOT processed by the e-graph optimizer for While statements
    // So skip this

    // Process the if-condition: i & 1 == 0
    let if_cond = Expr::BinOp {
        span: jules::compiler::lexer::Span::dummy(),
        op: BinOpKind::Eq,
        lhs: Box::new(Expr::BinOp {
            span: jules::compiler::lexer::Span::dummy(),
            op: BinOpKind::BitAnd,
            lhs: Box::new(Expr::Ident { span: jules::compiler::lexer::Span::dummy(), name: "i".into() }),
            rhs: Box::new(Expr::IntLit { span: jules::compiler::lexer::Span::dummy(), value: 1, ty: None }),
        }),
        rhs: Box::new(Expr::IntLit { span: jules::compiler::lexer::Span::dummy(), value: 0, ty: None }),
    };
    let if_cond_cls = eg.build_expr(&if_cond);
    println!("After i&1==0: if_cond -> class {}", if_cond_cls);

    // Saturate
    eg.saturate(4);

    // Extract best
    let best = eg.extract_best();
    let best_expr = eg.materialize(if_cond_cls, &best, jules::compiler::lexer::Span::dummy(), 64);
    println!("Best expression for if_cond: {:?}", best_expr);

    // Check if i & 1 and 0 are in the same e-class
    // We need to check the canonical classes
    // Let's rebuild to check
    let i_cls = eg.build_expr(&Expr::Ident { span: jules::compiler::lexer::Span::dummy(), name: "i".into() });
    let zero_cls = eg.build_expr(&Expr::IntLit { span: jules::compiler::lexer::Span::dummy(), value: 0, ty: None });
    let i_and_1 = eg.build_expr(&Expr::BinOp {
        span: jules::compiler::lexer::Span::dummy(),
        op: BinOpKind::BitAnd,
        lhs: Box::new(Expr::Ident { span: jules::compiler::lexer::Span::dummy(), name: "i".into() }),
        rhs: Box::new(Expr::IntLit { span: jules::compiler::lexer::Span::dummy(), value: 1, ty: None }),
    });
    println!("Var(i) class: {}, IntLit(0) class: {}, BitAnd(i,1) class: {}", i_cls, zero_cls, i_and_1);
}
