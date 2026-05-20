use jules::compiler::lower::lower_program;
use jules::compiler::parser::Parser;
use jules::compiler::lexer::Lexer;
use jules::compiler::typeck::TypeCk;

fn main() {
    let src = r#"fn main() -> i32 {
    let mut count: i32 = 0;
    let mut i: i32 = 0;
    while i < 200 {
        let a: i32 = i / 100;
        let b: i32 = (i / 10) - (i / 100) * 10;
        let c: i32 = i - (i / 10) * 10;
        if (a > b && b > c) || (a < b && b < c) {
            count = count + 1;
        }
        i = i + 1;
    }
    count
}"#;

    // Lex + parse + typecheck (AST level, needed for proper lowering)
    let mut lexer = Lexer::new(src);
    let (tokens, _) = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    let mut program = parser.parse_program();
    
    // Run AST type checker (needed for the program to be properly annotated)
    let mut typeck = TypeCk::new();
    typeck.check_program(&mut program);

    // Lower to IR
    let ir_module = lower_program(&program);

    println!("=== IR Module ===");
    for func in &ir_module.functions {
        println!("Function: {} (returns {:?})", func.name, func.ret_ty);
        for block in &func.blocks {
            println!("  bb{}:", block.id.0);
            for instr in &block.instrs {
                println!("    {:?}", instr);
            }
        }
    }

    // Now run type checker
    let typeck_result = jules::compiler::ir_typeck::ir_typeck(&ir_module);
    println!("\n=== IR Type Check Errors ===");
    for d in &typeck_result.diagnostics {
        println!("  {:?}: {}", d.kind, d.message);
    }
}
