// =============================================================================
// jules/src/parser.rs
//
// Recursive-descent parser for the Jules programming language.
//
// Input  : a flat Vec<Token> produced by the Lexer.
// Output : a fully-formed ast::Program, or one-or-more ParseErrors.
//
// Design notes
// ────────────
//  • Errors are accumulated rather than aborting on the first one.
//    `parse_program` always returns a (possibly-partial) Program alongside
//    any errors so downstream passes can still run and report more issues.
//  • Panic-recovery: after a syntax error the parser resynchronises to the
//    next statement boundary (`;` or `}`).
//  • Precedence climbing is used for binary expressions so the operator
//    table is easy to extend.
//  • Every `parse_*` method returns `ParseResult<T>` = `Result<T, ParseError>`.
//    Callers that want to continue past an error call `try_parse` helpers.
// =============================================================================

#![allow(dead_code)]

use std::fmt;

use crate::ast::*;
use crate::lexer::{Span, Token, TokenKind};

// =============================================================================
// §1  PARSE ERROR
// =============================================================================

#[derive(Debug, Clone)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
    /// Optional suggestion ("help: …").
    pub hint: Option<String>,
}

impl ParseError {
    pub fn new(span: Span, msg: impl Into<String>) -> Self {
        ParseError {
            span,
            message: msg.into(),
            hint: None,
        }
    }
    pub fn with_hint(mut self, h: impl Into<String>) -> Self {
        self.hint = Some(h.into());
        self
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at {}: {}", self.span, self.message)
    }
}

pub type ParseResult<T> = Result<T, ParseError>;

// =============================================================================
// §2  PARSER STATE
// =============================================================================

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Accumulated non-fatal errors (parser continues after recovery).
    pub errors: Vec<ParseError>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        // Small upfront reserve avoids first diagnostic allocation on recovery paths.
        Parser {
            tokens,
            pos: 0,
            errors: Vec::with_capacity(4),
        }
    }

    // ── Token access ──────────────────────────────────────────────────────────

    #[inline(always)]
    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    #[inline(always)]
    fn peek2(&self) -> &Token {
        let idx = (self.pos + 1).min(self.tokens.len() - 1);
        &self.tokens[idx]
    }

    #[inline(always)]
    fn current_span(&self) -> Span {
        self.peek().span
    }

    #[inline(always)]
    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos.min(self.tokens.len() - 1)];
        if !matches!(tok.kind, TokenKind::Eof) {
            self.pos += 1;
        }
        tok
    }

    #[inline(always)]
    fn at_eof(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    #[inline(always)]
    fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(kind)
    }

    /// True when the current token matches `kind` exactly (including payload
    /// for keyword tokens, which carry no payload anyway).
    #[inline(always)]
    fn is(&self, kind: &TokenKind) -> bool {
        &self.peek().kind == kind
    }

    #[inline(always)]
    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.is(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consume token of `kind` or push an error and return Err.
    fn expect(&mut self, kind: &TokenKind) -> ParseResult<Span> {
        if self.is(kind) {
            Ok(self.advance().span)
        } else {
            let found = self.peek();
            let expected = token_label(kind);
            let mut err = ParseError::new(
                found.span,
                format!("expected {expected}, but found `{}`", found.kind),
            );
            if let Some(hint) = expected_token_hint(kind, &found.kind) {
                err = err.with_hint(hint);
            }
            Err(err)
        }
    }

    fn unexpected(&self, msg: &str) -> ParseError {
        ParseError::new(
            self.current_span(),
            format!("{msg}, found `{}`", self.peek().kind),
        )
    }

    // ── Identifier helpers ────────────────────────────────────────────────────

    fn expect_ident(&mut self) -> ParseResult<(Span, String)> {
        let tok = self.peek().clone();
        match &tok.kind {
            TokenKind::Ident(s) => {
                self.advance();
                Ok((tok.span, s.clone()))
            }
            // Allow any keyword-as-identifier if the grammar position is
            // unambiguous (e.g. field names that happen to be keywords).
            _ => Err(ParseError::new(
                tok.span,
                format!("expected an identifier name, found `{}`", tok.kind),
            )
            .with_hint("use a valid name like `player`, `health`, or `main`")),
        }
    }

    /// Consume an identifier OR a keyword that can serve as a name here.
    fn expect_name(&mut self) -> ParseResult<(Span, String)> {
        let tok = self.peek().clone();
        let name = kw_as_ident(&tok.kind);
        if let Some(n) = name {
            self.advance();
            Ok((tok.span, n.to_owned()))
        } else if let TokenKind::Ident(s) = &tok.kind {
            self.advance();
            Ok((tok.span, s.clone()))
        } else {
            Err(
                ParseError::new(tok.span, format!("expected a name, found `{}`", tok.kind))
                    .with_hint("expected a declaration name here, for example `fn main`"),
            )
        }
    }

    // ── Error recovery ────────────────────────────────────────────────────────

    /// Push a non-fatal error and synchronise to the next statement/item boundary.
    fn recover(&mut self, e: ParseError) {
        self.errors.push(e);
        // Skip tokens until we see a plausible recovery point.
        loop {
            match self.peek().kind {
                TokenKind::Eof
                | TokenKind::Semicolon
                | TokenKind::KwFn
                | TokenKind::KwLet
                | TokenKind::KwReturn
                | TokenKind::KwIf
                | TokenKind::KwFor
                | TokenKind::KwWhile
                | TokenKind::KwLoop
                | TokenKind::KwSystem
                | TokenKind::KwComponent
                | TokenKind::KwAgent
                | TokenKind::KwModel
                | TokenKind::KwTrain
                | TokenKind::KwStruct
                | TokenKind::KwEnum => break,
                TokenKind::RBrace => {
                    self.advance();
                    break;
                }
                _ => {
                    self.advance();
                }
            }
        }
        self.eat(&TokenKind::Semicolon);
    }
}

fn token_label(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Semicolon => "`;` (statement terminator)".into(),
        TokenKind::Comma => "`,` (separator)".into(),
        TokenKind::Colon => "`:`".into(),
        TokenKind::LParen => "`(`".into(),
        TokenKind::RParen => "`)`".into(),
        TokenKind::LBrace => "`{`".into(),
        TokenKind::RBrace => "`}`".into(),
        TokenKind::LBracket => "`[`".into(),
        TokenKind::RBracket => "`]`".into(),
        TokenKind::Arrow => "`->`".into(),
        TokenKind::FatArrow => "`=>`".into(),
        TokenKind::Eq => "`=`".into(),
        _ => format!("`{kind}`"),
    }
}

fn expected_token_hint(expected: &TokenKind, found: &TokenKind) -> Option<String> {
    match expected {
        TokenKind::Semicolon => Some(match found {
            TokenKind::KwLet
            | TokenKind::KwFn
            | TokenKind::KwIf
            | TokenKind::KwFor
            | TokenKind::KwWhile
            | TokenKind::KwReturn => {
                "missing `;` before the next statement; add `;` to end this statement".into()
            }
            _ => "add `;` to end this statement".into(),
        }),
        TokenKind::RParen => Some("close this expression with `)`".into()),
        TokenKind::RBrace => Some("close this block with `}`".into()),
        TokenKind::RBracket => Some("close this index/type with `]`".into()),
        TokenKind::LBrace => Some("start a block with `{ ... }` after this declaration".into()),
        TokenKind::Comma => Some("separate items with `,`".into()),
        TokenKind::Arrow => Some("use `->` before a return type".into()),
        TokenKind::Eq => Some(match found {
            TokenKind::EqEq => "use `=` to assign a value (not `==`)".into(),
            _ => "use `=` to assign a value".into(),
        }),
        _ if matches!(found, TokenKind::Eof) => {
            Some("you may be missing code at the end of the file".into())
        }
        _ => None,
    }
}

/// Allow certain keywords to appear as identifier names in field/param positions.
fn kw_as_ident(kind: &TokenKind) -> Option<&'static str> {
    match kind {
        TokenKind::KwSelf => Some("self"),
        TokenKind::KwWorld => Some("world"),
        TokenKind::KwInput => Some("input"),
        TokenKind::KwOutput => Some("output"),
        TokenKind::KwLinear => Some("linear"),
        TokenKind::KwPolicy => Some("policy"),
        _ => None,
    }
}

// =============================================================================
// §3  TOP-LEVEL PROGRAM
// =============================================================================

impl Parser {
    /// Parse a complete source file into a `Program`.
    /// Always returns a Program (possibly empty) plus accumulated errors.
    pub fn parse_program(&mut self) -> Program {
        let start = self.current_span();
        let mut items = Vec::new();

        while !self.at_eof() {
            // Skip stray semicolons at top level.
            if self.eat(&TokenKind::Semicolon) {
                continue;
            }

            match self.parse_item() {
                Ok(item) => items.push(item),
                Err(e) => self.recover(e),
            }
        }

        let end = self.current_span();
        Program {
            span: start.merge(end),
            items,
        }
    }

    // ── Item dispatcher ───────────────────────────────────────────────────────

    fn parse_item(&mut self) -> ParseResult<Item> {
        // Collect leading attributes (@gpu, @grad, @inline …)
        let attrs = self.parse_attrs();

        // Visibility modifier
        let _pub = self.eat(&TokenKind::KwPub);

        match self.peek().kind.clone() {
            TokenKind::KwFn | TokenKind::KwAsync => self.parse_fn(attrs).map(Item::Fn),
            TokenKind::KwSystem => self.parse_system(attrs).map(Item::System),
            TokenKind::KwStruct => self.parse_struct(attrs).map(Item::Struct),
            TokenKind::KwComponent => self.parse_component(attrs).map(Item::Component),
            TokenKind::KwEnum => self.parse_enum(attrs).map(Item::Enum),
            TokenKind::KwConst => self.parse_const().map(Item::Const),
            TokenKind::KwUse => self.parse_use().map(Item::Use),
            TokenKind::KwMod => self.parse_mod(_pub),
            TokenKind::KwAgent => self.parse_agent(attrs).map(Item::Agent),
            TokenKind::KwModel => self.parse_model(attrs).map(Item::Model),
            TokenKind::KwTrain => self.parse_train(attrs).map(Item::Train),
            TokenKind::KwShader => self.parse_shader(attrs).map(Item::Shader),
            TokenKind::KwScene => self.parse_scene(attrs).map(Item::Scene),
            TokenKind::KwPrefab => self.parse_prefab(attrs).map(Item::Prefab),
            TokenKind::KwPhysics => self.parse_physics_config(attrs).map(Item::PhysicsConfig),
            TokenKind::KwLoss => self.parse_loss(attrs).map(Item::Loss),
            _ => Err(self.unexpected("expected item declaration")),
        }
    }

    // ── Attributes ────────────────────────────────────────────────────────────

    fn parse_attrs(&mut self) -> Vec<Attribute> {
        let mut attrs = Vec::new();
        loop {
            let attr = match &self.peek().kind {
                TokenKind::AtGpu => {
                    self.advance();
                    Attribute::Gpu
                }
                TokenKind::AtCpu => {
                    self.advance();
                    Attribute::Cpu
                }
                TokenKind::AtTpu => {
                    self.advance();
                    Attribute::Tpu
                }
                TokenKind::AtGrad => {
                    self.advance();
                    Attribute::Grad
                }
                TokenKind::AtSimd => {
                    self.advance();
                    Attribute::Simd
                }
                TokenKind::AtParallel => {
                    self.advance();
                    Attribute::Parallel
                }
                TokenKind::AtSeq => {
                    self.advance();
                    Attribute::Seq
                }
                // Named annotations with optional args
                TokenKind::AtKernel
                | TokenKind::AtInline
                | TokenKind::AtNoInline
                | TokenKind::AtUnroll
                | TokenKind::AtJit
                | TokenKind::AtVmap
                | TokenKind::AtCheckpoint
                | TokenKind::AtQuantize
                | TokenKind::AtPrune
                | TokenKind::AtProfile
                | TokenKind::AtTest
                | TokenKind::AtBenchmark
                | TokenKind::AtDeprecated
                | TokenKind::AtAi => {
                    let raw = format!("{:?}", self.peek().kind);
                    let name = raw.trim_start_matches("At").to_lowercase();
                    self.advance();
                    let args = if self.is(&TokenKind::LParen) {
                        self.parse_named_attr_args().unwrap_or_default()
                    } else {
                        vec![]
                    };
                    Attribute::Named { name, args }
                }
                TokenKind::AtCustom(custom_name) => {
                    let name = custom_name.to_lowercase();
                    self.advance();
                    let args = if self.is(&TokenKind::LParen) {
                        self.parse_named_attr_args().unwrap_or_default()
                    } else {
                        vec![]
                    };
                    Attribute::Named { name, args }
                }
                _ => break,
            };
            attrs.push(attr);
        }
        attrs
    }

    fn parse_named_attr_args(&mut self) -> ParseResult<Vec<Expr>> {
        self.expect(&TokenKind::LParen)?;
        let mut args = Vec::new();

        if self.eat(&TokenKind::RParen) {
            return Ok(args);
        }

        // Support architecture shorthand as first positional arg:
        //   @PPO(124->248->68, lr = 0.0003)
        if let Some(expr) = self.try_parse_architecture_chain_expr()? {
            args.push(expr);
        } else {
            // Keep parity with call args: support `name: value` as a shorthand.
            if matches!(&self.peek().kind, TokenKind::Ident(_))
                && self.peek2().kind == TokenKind::Colon
            {
                self.advance();
                self.advance();
            }
            args.push(self.parse_expr(0)?);
        }

        while self.eat(&TokenKind::Comma) {
            if self.is(&TokenKind::RParen) {
                break;
            }
            if matches!(&self.peek().kind, TokenKind::Ident(_))
                && self.peek2().kind == TokenKind::Colon
            {
                self.advance();
                self.advance();
            }
            args.push(self.parse_expr(0)?);
        }

        self.expect(&TokenKind::RParen)?;
        Ok(args)
    }

    fn try_parse_architecture_chain_expr(&mut self) -> ParseResult<Option<Expr>> {
        let save = self.pos;
        let mut parts = Vec::new();
        let mut saw_arrow = false;
        loop {
            match &self.peek().kind {
                TokenKind::IntLit { value, .. } => {
                    parts.push(value.to_string());
                    self.advance();
                }
                TokenKind::Ident(name) => {
                    parts.push(name.clone());
                    self.advance();
                }
                _ => {
                    self.pos = save;
                    return Ok(None);
                }
            }
            if self.eat(&TokenKind::Arrow) {
                saw_arrow = true;
                parts.push("->".to_string());
                continue;
            }
            break;
        }

        if !saw_arrow {
            self.pos = save;
            return Ok(None);
        }

        let span = self.tokens[save]
            .span
            .merge(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Some(Expr::StrLit {
            span,
            value: parts.join(""),
        }))
    }
}

// =============================================================================
// §4  FUNCTION + SYSTEM
// =============================================================================

impl Parser {
    fn stmt_if_to_expr(stmt: &Stmt) -> Option<Expr> {
        match stmt {
            Stmt::If {
                span,
                cond,
                then,
                else_,
            } => {
                let else_ = else_.as_ref().and_then(|branch| match branch.as_ref() {
                    IfOrBlock::Block(b) => Some(Box::new(b.clone())),
                    IfOrBlock::If(inner) => {
                        let nested = Self::stmt_if_to_expr(inner)?;
                        Some(Box::new(Block {
                            span: nested.span(),
                            stmts: vec![],
                            tail: Some(Box::new(nested)),
                        }))
                    }
                });
                Some(Expr::IfExpr {
                    span: *span,
                    cond: Box::new(cond.clone()),
                    then: Box::new(then.clone()),
                    else_,
                })
            }
            _ => None,
        }
    }

    fn parse_fn(&mut self, attrs: Vec<Attribute>) -> ParseResult<FnDecl> {
        let start = self.current_span();
        let is_async = self.eat(&TokenKind::KwAsync);
        self.expect(&TokenKind::KwFn)?;
        let (_, name) = self.expect_ident()?;
        let generics = self.parse_generics()?;
        let params = self.parse_params()?;

        let ret_ty = if self.eat(&TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = if self.is(&TokenKind::LBrace) {
            let mut body = self.parse_block()?;
            if ret_ty.is_some() && body.tail.is_none() {
                if let Some(stmt) = body.stmts.last().cloned() {
                    if let Some(expr) = Self::stmt_if_to_expr(&stmt) {
                        body.stmts.pop();
                        body.tail = Some(Box::new(expr));
                    }
                }
            }
            Some(body)
        } else {
            self.eat(&TokenKind::Semicolon);
            None
        };

        Ok(FnDecl {
            span: start.merge(self.current_span()),
            attrs,
            name,
            generics,
            params,
            ret_ty,
            body,
            is_async,
        })
    }

    fn parse_system(&mut self, attrs: Vec<Attribute>) -> ParseResult<SystemDecl> {
        let start = self.current_span();
        self.expect(&TokenKind::KwSystem)?;
        let (_, name) = self.expect_ident()?;
        let params = self.parse_params()?;

        // Optional explicit query annotation: `query(Pos, Vel) without(Dead)`
        let explicit_query = if self.eat(&TokenKind::KwQuery) {
            Some(self.parse_entity_query_clause(start)?)
        } else {
            None
        };

        let body = self.parse_block()?;
        let span = start.merge(self.current_span());
        Ok(SystemDecl {
            span,
            attrs,
            name,
            params,
            explicit_query,
            body,
            accesses: vec![],
            parallelism: ParallelismHint::Auto,
            iterations_independent: false,
        })
    }

    fn parse_generics(&mut self) -> ParseResult<Vec<GenericParam>> {
        if !self.is(&TokenKind::Lt) {
            return Ok(vec![]);
        }
        self.advance(); // <
        let mut params = Vec::new();
        while !self.is(&TokenKind::Gt) && !self.at_eof() {
            let span = self.current_span();
            let (_, name) = self.expect_ident()?;
            let mut bounds = Vec::new();
            if self.eat(&TokenKind::Colon) {
                bounds.push(self.expect_ident()?.1);
                while self.eat(&TokenKind::Plus) {
                    bounds.push(self.expect_ident()?.1);
                }
            }
            params.push(GenericParam { span, name, bounds });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::Gt)?;
        Ok(params)
    }

    fn parse_params(&mut self) -> ParseResult<Vec<Param>> {
        self.expect(&TokenKind::LParen)?;
        let mut params = Vec::new();
        while !self.is(&TokenKind::RParen) && !self.at_eof() {
            let param = self.parse_param()?;
            params.push(param);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok(params)
    }

    fn parse_param(&mut self) -> ParseResult<Param> {
        let span = self.current_span();
        let mutable = self.eat(&TokenKind::KwMut);
        let (_, name) = self.expect_name()?;
        let ty = if self.eat(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        let default = if self.eat(&TokenKind::Eq) {
            Some(self.parse_expr(0)?)
        } else {
            None
        };
        Ok(Param {
            span,
            name,
            ty,
            default,
            mutable,
        })
    }
}

// =============================================================================
// §5  STRUCT / COMPONENT / ENUM / CONST / USE / MOD
// =============================================================================

impl Parser {
    fn parse_struct(&mut self, attrs: Vec<Attribute>) -> ParseResult<StructDecl> {
        let start = self.current_span();
        self.expect(&TokenKind::KwStruct)?;
        let (_, name) = self.expect_ident()?;
        let generics = self.parse_generics()?;
        let fields = self.parse_struct_fields()?;
        Ok(StructDecl {
            span: start.merge(self.current_span()),
            attrs,
            name,
            generics,
            fields,
        })
    }

    fn parse_component(&mut self, attrs: Vec<Attribute>) -> ParseResult<ComponentDecl> {
        let start = self.current_span();
        self.expect(&TokenKind::KwComponent)?;
        let (_, name) = self.expect_ident()?;
        let fields = self.parse_struct_fields()?;
        Ok(ComponentDecl {
            span: start.merge(self.current_span()),
            attrs,
            name,
            fields,
            layout: ComponentLayout::Soa,
        })
    }

    fn parse_struct_fields(&mut self) -> ParseResult<Vec<StructField>> {
        self.expect(&TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            let span = self.current_span();
            let fattrs = self.parse_attrs();
            let (_, fname) = self.expect_name()?;
            self.expect(&TokenKind::Colon)?;
            let ftype = self.parse_type()?;
            fields.push(StructField {
                span,
                name: fname,
                ty: ftype,
                attrs: fattrs,
            });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(fields)
    }

    fn parse_enum(&mut self, attrs: Vec<Attribute>) -> ParseResult<EnumDecl> {
        let start = self.current_span();
        self.expect(&TokenKind::KwEnum)?;
        let (_, name) = self.expect_ident()?;
        let generics = self.parse_generics()?;
        self.expect(&TokenKind::LBrace)?;
        let mut variants = Vec::new();
        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            let span = self.current_span();
            let (_, vname) = self.expect_ident()?;
            let fields = if self.is(&TokenKind::LBrace) {
                let sf = self.parse_struct_fields()?;
                EnumVariantFields::Struct(sf)
            } else if self.is(&TokenKind::LParen) {
                self.advance();
                let mut types = Vec::new();
                while !self.is(&TokenKind::RParen) && !self.at_eof() {
                    types.push(self.parse_type()?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen)?;
                EnumVariantFields::Tuple(types)
            } else {
                EnumVariantFields::Unit
            };
            variants.push(EnumVariant {
                span,
                name: vname,
                fields,
            });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(EnumDecl {
            span: start.merge(self.current_span()),
            attrs,
            name,
            generics,
            variants,
        })
    }

    fn parse_const(&mut self) -> ParseResult<ConstDecl> {
        let start = self.current_span();
        let is_pub = self.peek().kind == TokenKind::KwPub; // already consumed above
        self.expect(&TokenKind::KwConst)?;
        let (_, name) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr(0)?;
        self.eat(&TokenKind::Semicolon);
        Ok(ConstDecl {
            span: start.merge(self.current_span()),
            name,
            ty,
            value,
            is_pub,
        })
    }

    fn parse_use(&mut self) -> ParseResult<UsePath> {
        let start = self.current_span();
        self.expect(&TokenKind::KwUse)?;
        let mut segments = Vec::new();
        loop {
            let (_, seg) = self.expect_name()?;
            segments.push(seg);
            if !self.eat(&TokenKind::ColonColon) {
                break;
            }
            if self.is(&TokenKind::Star) {
                self.advance();
                break;
            }
        }
        let is_glob = segments.last().map(|s| s == "*").unwrap_or(false);
        let alias = if self.eat(&TokenKind::KwAs) {
            Some(self.expect_ident()?.1)
        } else {
            None
        };
        self.eat(&TokenKind::Semicolon);
        Ok(UsePath {
            span: start.merge(self.current_span()),
            segments,
            alias,
            is_glob,
        })
    }

    fn parse_mod(&mut self, is_pub: bool) -> ParseResult<Item> {
        let start = self.current_span();
        self.expect(&TokenKind::KwMod)?;
        let (_, name) = self.expect_ident()?;
        let items = if self.is(&TokenKind::LBrace) {
            self.advance();
            let mut inner = Vec::new();
            while !self.is(&TokenKind::RBrace) && !self.at_eof() {
                match self.parse_item() {
                    Ok(i) => inner.push(i),
                    Err(e) => self.recover(e),
                }
            }
            self.expect(&TokenKind::RBrace)?;
            Some(inner)
        } else {
            self.eat(&TokenKind::Semicolon);
            None
        };
        Ok(Item::Mod {
            span: start.merge(self.current_span()),
            name,
            items,
            is_pub,
        })
    }
}

// =============================================================================
// §6  TYPES
// =============================================================================

impl Parser {
    pub fn parse_type(&mut self) -> ParseResult<Type> {
        match self.peek().kind.clone() {
            // Scalars / primitive types
            TokenKind::KwF16 => {
                self.advance();
                Ok(Type::Scalar(ElemType::F16))
            }
            TokenKind::KwF32 => {
                self.advance();
                Ok(Type::Scalar(ElemType::F32))
            }
            TokenKind::KwF64 => {
                self.advance();
                Ok(Type::Scalar(ElemType::F64))
            }
            TokenKind::KwBf16 => {
                self.advance();
                Ok(Type::Scalar(ElemType::Bf16))
            }
            TokenKind::KwI8 => {
                self.advance();
                Ok(Type::Scalar(ElemType::I8))
            }
            TokenKind::KwI16 => {
                self.advance();
                Ok(Type::Scalar(ElemType::I16))
            }
            TokenKind::KwI32 => {
                self.advance();
                Ok(Type::Scalar(ElemType::I32))
            }
            TokenKind::KwI64 => {
                self.advance();
                Ok(Type::Scalar(ElemType::I64))
            }
            TokenKind::KwU8 => {
                self.advance();
                Ok(Type::Scalar(ElemType::U8))
            }
            TokenKind::KwU16 => {
                self.advance();
                Ok(Type::Scalar(ElemType::U16))
            }
            TokenKind::KwU32 => {
                self.advance();
                Ok(Type::Scalar(ElemType::U32))
            }
            TokenKind::KwU64 => {
                self.advance();
                Ok(Type::Scalar(ElemType::U64))
            }
            TokenKind::KwBool => {
                self.advance();
                Ok(Type::Scalar(ElemType::Bool))
            }
            TokenKind::KwUsize => {
                self.advance();
                Ok(Type::Scalar(ElemType::Usize))
            }

            // Tensor<elem>[dims…]
            TokenKind::KwTensor => self.parse_tensor_type(),

            // SIMD vectors / matrices
            TokenKind::KwVec2 => {
                self.advance();
                Ok(Type::Vec {
                    size: VecSize::N2,
                    family: VecFamily::Float,
                })
            }
            TokenKind::KwVec3 => {
                self.advance();
                Ok(Type::Vec {
                    size: VecSize::N3,
                    family: VecFamily::Float,
                })
            }
            TokenKind::KwVec4 => {
                self.advance();
                Ok(Type::Vec {
                    size: VecSize::N4,
                    family: VecFamily::Float,
                })
            }
            TokenKind::KwIVec2 => {
                self.advance();
                Ok(Type::Vec {
                    size: VecSize::N2,
                    family: VecFamily::Int,
                })
            }
            TokenKind::KwIVec3 => {
                self.advance();
                Ok(Type::Vec {
                    size: VecSize::N3,
                    family: VecFamily::Int,
                })
            }
            TokenKind::KwIVec4 => {
                self.advance();
                Ok(Type::Vec {
                    size: VecSize::N4,
                    family: VecFamily::Int,
                })
            }
            TokenKind::KwUVec2 => {
                self.advance();
                Ok(Type::Vec {
                    size: VecSize::N2,
                    family: VecFamily::UInt,
                })
            }
            TokenKind::KwUVec3 => {
                self.advance();
                Ok(Type::Vec {
                    size: VecSize::N3,
                    family: VecFamily::UInt,
                })
            }
            TokenKind::KwUVec4 => {
                self.advance();
                Ok(Type::Vec {
                    size: VecSize::N4,
                    family: VecFamily::UInt,
                })
            }
            TokenKind::KwMat2 => {
                self.advance();
                Ok(Type::Mat { size: VecSize::N2 })
            }
            TokenKind::KwMat3 => {
                self.advance();
                Ok(Type::Mat { size: VecSize::N3 })
            }
            TokenKind::KwMat4 => {
                self.advance();
                Ok(Type::Mat { size: VecSize::N4 })
            }
            TokenKind::KwQuat => {
                self.advance();
                Ok(Type::Quat)
            }

            // Tuple type: (A, B, C)
            TokenKind::LParen => {
                let _start = self.advance().span;
                if self.eat(&TokenKind::RParen) {
                    return Ok(Type::Tuple(vec![])); // unit ()
                }
                let first = self.parse_type()?;
                if self.eat(&TokenKind::RParen) {
                    return Ok(first); // parenthesised type, not tuple
                }
                let mut types = vec![first];
                while self.eat(&TokenKind::Comma) {
                    if self.is(&TokenKind::RParen) {
                        break;
                    }
                    types.push(self.parse_type()?);
                }
                self.expect(&TokenKind::RParen)?;
                Ok(Type::Tuple(types))
            }

            // Slice / array: [T] or [T; N]
            TokenKind::LBracket => {
                self.advance();
                let inner = self.parse_type()?;
                if self.eat(&TokenKind::Semicolon) {
                    let len_expr = self.parse_expr(0)?;
                    self.expect(&TokenKind::RBracket)?;
                    Ok(Type::Array {
                        elem: Box::new(inner),
                        len: Box::new(len_expr),
                    })
                } else {
                    self.expect(&TokenKind::RBracket)?;
                    Ok(Type::Slice(Box::new(inner)))
                }
            }

            // Reference: &T or &mut T
            TokenKind::Ampersand => {
                self.advance();
                let mutable = self.eat(&TokenKind::KwMut);
                let inner = self.parse_type()?;
                Ok(Type::Ref {
                    mutable,
                    inner: Box::new(inner),
                })
            }

            // Never type: !
            TokenKind::Bang => {
                self.advance();
                Ok(Type::Never)
            }

            // Infer: _
            TokenKind::Ident(s) if s == "_" => {
                self.advance();
                Ok(Type::Infer)
            }

            // fn(A, B) -> C
            TokenKind::KwFn => {
                self.advance();
                self.expect(&TokenKind::LParen)?;
                let mut params = Vec::new();
                while !self.is(&TokenKind::RParen) && !self.at_eof() {
                    params.push(self.parse_type()?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen)?;
                let ret = if self.eat(&TokenKind::Arrow) {
                    self.parse_type()?
                } else {
                    Type::Tuple(vec![])
                };
                Ok(Type::FnPtr {
                    params,
                    ret: Box::new(ret),
                })
            }

            // Named type or Option<T>
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();
                // Option<T>
                if name == "Option" && self.eat(&TokenKind::Lt) {
                    let inner = self.parse_type()?;
                    self.expect(&TokenKind::Gt)?;
                    return Ok(Type::Option(Box::new(inner)));
                }
                Ok(Type::Named(name))
            }

            other => Err(ParseError::new(
                self.current_span(),
                format!("expected type, found `{other:?}`"),
            )),
        }
    }

    fn parse_tensor_type(&mut self) -> ParseResult<Type> {
        self.expect(&TokenKind::KwTensor)?;
        // tensor<elem>
        self.expect(&TokenKind::Lt)?;
        let elem = self.parse_elem_type()?;
        self.expect(&TokenKind::Gt)?;
        // [dims…]
        self.expect(&TokenKind::LBracket)?;
        let mut shape = Vec::new();
        while !self.is(&TokenKind::RBracket) && !self.at_eof() {
            let dim = match &self.peek().kind.clone() {
                TokenKind::IntLit { value, .. } => {
                    let v = *value as u64;
                    self.advance();
                    DimExpr::Lit(v)
                }
                TokenKind::Ident(s) if s == "_" => {
                    self.advance();
                    DimExpr::Dynamic
                }
                TokenKind::Ident(s) => {
                    let name = s.clone();
                    self.advance();
                    DimExpr::Named(name)
                }
                _ => {
                    let e = self.parse_expr(0)?;
                    DimExpr::Expr(Box::new(e))
                }
            };
            shape.push(dim);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RBracket)?;
        Ok(Type::Tensor { elem, shape })
    }

    fn parse_elem_type(&mut self) -> ParseResult<ElemType> {
        match self.peek().kind.clone() {
            TokenKind::KwF16 => {
                self.advance();
                Ok(ElemType::F16)
            }
            TokenKind::KwF32 => {
                self.advance();
                Ok(ElemType::F32)
            }
            TokenKind::KwF64 => {
                self.advance();
                Ok(ElemType::F64)
            }
            TokenKind::KwBf16 => {
                self.advance();
                Ok(ElemType::Bf16)
            }
            TokenKind::KwI8 => {
                self.advance();
                Ok(ElemType::I8)
            }
            TokenKind::KwI16 => {
                self.advance();
                Ok(ElemType::I16)
            }
            TokenKind::KwI32 => {
                self.advance();
                Ok(ElemType::I32)
            }
            TokenKind::KwI64 => {
                self.advance();
                Ok(ElemType::I64)
            }
            TokenKind::KwU8 => {
                self.advance();
                Ok(ElemType::U8)
            }
            TokenKind::KwU16 => {
                self.advance();
                Ok(ElemType::U16)
            }
            TokenKind::KwU32 => {
                self.advance();
                Ok(ElemType::U32)
            }
            TokenKind::KwU64 => {
                self.advance();
                Ok(ElemType::U64)
            }
            TokenKind::KwBool => {
                self.advance();
                Ok(ElemType::Bool)
            }
            TokenKind::KwUsize => {
                self.advance();
                Ok(ElemType::Usize)
            }
            other => Err(ParseError::new(
                self.current_span(),
                format!("expected element type inside tensor<…>, found `{other:?}`"),
            )),
        }
    }
}

// =============================================================================
// §7  BLOCKS AND STATEMENTS
// =============================================================================

impl Parser {
    fn parse_block(&mut self) -> ParseResult<Block> {
        let start = self.expect(&TokenKind::LBrace)?;
        let mut stmts = Vec::new();

        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            // Skip stray semicolons.
            if self.eat(&TokenKind::Semicolon) {
                continue;
            }

            match self.parse_stmt() {
                Ok(stmt) => stmts.push(stmt),
                Err(e) => self.recover(e),
            }
        }

        let end = self.expect(&TokenKind::RBrace)?;

        // Detect trailing expression (no semicolon before `}`).
        let tail = if let Some(Stmt::Expr {
            expr,
            has_semi: false,
            ..
        }) = stmts.last().cloned()
        {
            stmts.pop();
            Some(Box::new(expr))
        } else {
            None
        };

        Ok(Block {
            span: start.merge(end),
            stmts,
            tail,
        })
    }

    fn parse_stmt(&mut self) -> ParseResult<Stmt> {
        match self.peek().kind.clone() {
            // ── let binding ───────────────────────────────────────────────────
            TokenKind::KwLet => {
                let start = self.advance().span;
                let mutable = self.eat(&TokenKind::KwMut);
                let pattern = self.parse_pattern()?;
                let ty = if self.eat(&TokenKind::Colon) {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                let init = if self.eat(&TokenKind::Eq) {
                    Some(self.parse_expr(0)?)
                } else {
                    None
                };
                self.eat(&TokenKind::Semicolon);
                Ok(Stmt::Let {
                    span: start,
                    pattern,
                    ty,
                    init,
                    mutable,
                })
            }

            // ── return ────────────────────────────────────────────────────────
            TokenKind::KwReturn => {
                let span = self.advance().span;
                let value = if !self.is(&TokenKind::Semicolon) && !self.is(&TokenKind::RBrace) {
                    Some(self.parse_expr(0)?)
                } else {
                    None
                };
                self.eat(&TokenKind::Semicolon);
                Ok(Stmt::Return { span, value })
            }

            // ── break ─────────────────────────────────────────────────────────
            TokenKind::KwBreak => {
                let span = self.advance().span;
                let label = self.try_parse_label();
                let value = if !self.is(&TokenKind::Semicolon) && !self.is(&TokenKind::RBrace) {
                    Some(self.parse_expr(0)?)
                } else {
                    None
                };
                self.eat(&TokenKind::Semicolon);
                Ok(Stmt::Break { span, value, label })
            }

            // ── continue ──────────────────────────────────────────────────────
            TokenKind::KwContinue => {
                let span = self.advance().span;
                let label = self.try_parse_label();
                self.eat(&TokenKind::Semicolon);
                Ok(Stmt::Continue { span, label })
            }

            // ── for / entity-for ──────────────────────────────────────────────
            TokenKind::KwFor => self.parse_for_stmt(),

            // ── parallel for ──────────────────────────────────────────────────
            TokenKind::KwParallel => self.parse_parallel_for().map(Stmt::ParallelFor),

            // ── while ─────────────────────────────────────────────────────────
            TokenKind::KwWhile => {
                let span = self.advance().span;
                let label = self.try_parse_label();
                let cond = self.parse_expr(0)?;
                let body = self.parse_block()?;
                Ok(Stmt::While {
                    span,
                    cond,
                    body,
                    label,
                })
            }

            // ── loop ──────────────────────────────────────────────────────────
            TokenKind::KwLoop => {
                let span = self.advance().span;
                let label = self.try_parse_label();
                let body = self.parse_block()?;
                Ok(Stmt::Loop { span, body, label })
            }

            // ── if ────────────────────────────────────────────────────────────
            TokenKind::KwIf => self.parse_if_stmt(),

            // ── match ─────────────────────────────────────────────────────────
            TokenKind::KwMatch => self.parse_match_stmt(),

            // ── spawn / sync / atomic ─────────────────────────────────────────
            TokenKind::KwSpawn => self.parse_spawn().map(Stmt::Spawn),
            TokenKind::KwSync => self.parse_sync().map(Stmt::Sync),
            TokenKind::KwAtomic => self.parse_atomic().map(Stmt::Atomic),

            // ── nested item ───────────────────────────────────────────────────
            TokenKind::KwFn
            | TokenKind::KwStruct
            | TokenKind::KwEnum
            | TokenKind::KwConst
            | TokenKind::KwComponent => {
                let item = self.parse_item()?;
                Ok(Stmt::Item(Box::new(item)))
            }

            // ── expression statement ──────────────────────────────────────────
            _ => {
                let span = self.current_span();
                let expr = self.parse_expr(0)?;
                let has_semi = self.eat(&TokenKind::Semicolon);
                Ok(Stmt::Expr {
                    span,
                    expr,
                    has_semi,
                })
            }
        }
    }

    fn try_parse_label(&mut self) -> Option<String> {
        // Labels: `'name:` written as a tick-prefixed ident — not in the
        // current lexer, so we leave this as None for now.
        None
    }

    fn parse_for_stmt(&mut self) -> ParseResult<Stmt> {
        let span = self.expect(&TokenKind::KwFor)?;
        let pattern = self.parse_pattern()?;
        self.expect(&TokenKind::KwIn)?;

        // Detect `for entity in world [with (…)] [without (…)]`
        let is_world_iter = self.is(&TokenKind::KwWorld) || {
            // also handle bare `world` identifier
            matches!(&self.peek().kind, TokenKind::Ident(s) if s == "world")
        };

        if is_world_iter {
            self.advance(); // consume `world`
            let query = self.parse_entity_query_clause(span)?;
            let body = self.parse_block()?;
            let var = if let Pattern::Ident { name, .. } = &pattern {
                name.clone()
            } else {
                "entity".into()
            };
            return Ok(Stmt::EntityFor {
                span,
                var,
                query,
                body,
                label: None,
                accesses: vec![],
                parallelism: ParallelismHint::Auto,
            });
        }

        let iter = self.parse_expr(0)?;
        let body = self.parse_block()?;
        Ok(Stmt::ForIn {
            span,
            pattern,
            iter,
            body,
            label: None,
        })
    }

    fn parse_entity_query_clause(&mut self, span: Span) -> ParseResult<EntityQuery> {
        let mut with = Vec::new();
        let mut without = Vec::new();

        if self.eat(&TokenKind::KwWith) {
            self.expect(&TokenKind::LParen)?;
            while !self.is(&TokenKind::RParen) && !self.at_eof() {
                with.push(self.expect_ident()?.1);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RParen)?;
        }
        if self.eat(&TokenKind::KwWithout) {
            self.expect(&TokenKind::LParen)?;
            while !self.is(&TokenKind::RParen) && !self.at_eof() {
                without.push(self.expect_ident()?.1);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RParen)?;
        }

        Ok(EntityQuery {
            span,
            with,
            without,
            filter: None,
        })
    }

    fn parse_if_stmt(&mut self) -> ParseResult<Stmt> {
        let span = self.expect(&TokenKind::KwIf)?;
        let cond = self.parse_expr(0)?;
        let then = self.parse_block()?;
        let else_ = if self.eat(&TokenKind::KwElse) {
            if self.is(&TokenKind::KwIf) {
                let inner = self.parse_if_stmt()?;
                Some(Box::new(IfOrBlock::If(inner)))
            } else {
                let b = self.parse_block()?;
                Some(Box::new(IfOrBlock::Block(b)))
            }
        } else {
            None
        };
        Ok(Stmt::If {
            span,
            cond,
            then,
            else_,
        })
    }

    fn parse_match_stmt(&mut self) -> ParseResult<Stmt> {
        let span = self.expect(&TokenKind::KwMatch)?;
        let expr = self.parse_expr(0)?;
        self.expect(&TokenKind::LBrace)?;
        let mut arms = Vec::new();
        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            let arm_span = self.current_span();
            let pat = self.parse_pattern()?;
            let guard = if self.eat(&TokenKind::KwIf) {
                Some(self.parse_expr(0)?)
            } else {
                None
            };
            self.expect(&TokenKind::FatArrow)?;
            let body = if self.is(&TokenKind::LBrace) {
                let b = self.parse_block()?;
                Expr::Block(Box::new(b))
            } else {
                self.parse_expr(0)?
            };
            self.eat(&TokenKind::Comma);
            arms.push(MatchArm {
                span: arm_span,
                pat,
                guard,
                body,
            });
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(Stmt::Match { span, expr, arms })
    }

    fn parse_parallel_for(&mut self) -> ParseResult<ParallelFor> {
        let span = self.expect(&TokenKind::KwParallel)?;
        // Optional (chunk=N)
        let chunk = if self.is(&TokenKind::LParen) {
            self.advance();
            // chunk = 64
            let _key = self.expect_ident()?.1;
            self.expect(&TokenKind::Eq)?;
            let n = match &self.peek().kind.clone() {
                TokenKind::IntLit { value, .. } => {
                    let v = *value as u64;
                    self.advance();
                    Some(v)
                }
                _ => None,
            };
            self.expect(&TokenKind::RParen)?;
            n
        } else {
            None
        };

        let attrs = self.parse_attrs();
        self.expect(&TokenKind::KwFor)?;
        let var = self.parse_pattern()?;
        self.expect(&TokenKind::KwIn)?;
        let iter = self.parse_expr(0)?;
        let body = self.parse_block()?;

        let schedule = schedule_from_attrs(&attrs);
        Ok(ParallelFor {
            span,
            attrs,
            var,
            iter,
            body,
            label: None,
            chunk,
            schedule,
        })
    }

    fn parse_spawn(&mut self) -> ParseResult<SpawnBlock> {
        let span = self.expect(&TokenKind::KwSpawn)?;
        let name = if self.is(&TokenKind::LParen) {
            self.advance();
            let n = match &self.peek().kind.clone() {
                TokenKind::StringLit { value } => {
                    let v = value.clone();
                    self.advance();
                    Some(v)
                }
                _ => None,
            };
            self.expect(&TokenKind::RParen)?;
            n
        } else {
            None
        };
        let body = self.parse_block()?;
        Ok(SpawnBlock {
            span,
            attrs: vec![],
            body,
            name,
        })
    }

    fn parse_sync(&mut self) -> ParseResult<SyncBlock> {
        let span = self.expect(&TokenKind::KwSync)?;
        let body = self.parse_block()?;
        Ok(SyncBlock { span, body })
    }

    fn parse_atomic(&mut self) -> ParseResult<AtomicBlock> {
        let span = self.expect(&TokenKind::KwAtomic)?;
        let body = self.parse_block()?;
        Ok(AtomicBlock { span, body })
    }
}

fn schedule_from_attrs(attrs: &[Attribute]) -> ScheduleKind {
    for a in attrs {
        match a {
            Attribute::Gpu => return ScheduleKind::Gpu,
            Attribute::Simd => return ScheduleKind::Simd,
            Attribute::Seq => return ScheduleKind::Sequential,
            Attribute::Parallel => return ScheduleKind::ThreadPool,
            _ => {}
        }
    }
    ScheduleKind::Auto
}

// =============================================================================
// §8  PATTERNS
// =============================================================================

impl Parser {
    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        let lhs = self.parse_pattern_atom()?;
        // Or-pattern: `a | b | c`
        if self.is(&TokenKind::Pipe) {
            let span = lhs.span();
            let mut arms = vec![lhs];
            while self.eat(&TokenKind::Pipe) {
                arms.push(self.parse_pattern_atom()?);
            }
            return Ok(Pattern::Or { span, arms });
        }
        Ok(lhs)
    }

    fn parse_pattern_atom(&mut self) -> ParseResult<Pattern> {
        match self.peek().kind.clone() {
            TokenKind::Ident(s) if s == "_" => Ok(Pattern::Wildcard(self.advance().span)),
            TokenKind::KwMut => {
                let span = self.advance().span;
                let (_, name) = self.expect_ident()?;
                Ok(Pattern::Ident {
                    span,
                    name,
                    mutable: true,
                })
            }
            TokenKind::IntLit { value, .. } => {
                let span = self.advance().span;
                Ok(Pattern::Lit(span, LitVal::Int(value)))
            }
            TokenKind::FloatLit { value } => {
                let span = self.advance().span;
                Ok(Pattern::Lit(span, LitVal::Float(value)))
            }
            TokenKind::BoolLit { value } => {
                let span = self.advance().span;
                Ok(Pattern::Lit(span, LitVal::Bool(value)))
            }
            TokenKind::StringLit { value } => {
                let span = self.advance().span;
                Ok(Pattern::Lit(span, LitVal::Str(value)))
            }
            // Tuple pattern: (a, b)
            TokenKind::LParen => {
                let span = self.advance().span;
                let mut elems = Vec::new();
                while !self.is(&TokenKind::RParen) && !self.at_eof() {
                    elems.push(self.parse_pattern()?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen)?;
                Ok(Pattern::Tuple { span, elems })
            }
            // Named pattern: Foo, Foo(x), Foo { x, y }
            TokenKind::Ident(name) => {
                let span = self.current_span();
                let name = name.clone();
                self.advance();

                // Path separator
                let mut path = name;
                while self.eat(&TokenKind::ColonColon) {
                    path.push_str("::");
                    path.push_str(&self.expect_ident()?.1);
                }

                if self.is(&TokenKind::LParen) {
                    // Enum variant: Some(x)
                    self.advance();
                    let mut inner = Vec::new();
                    while !self.is(&TokenKind::RParen) && !self.at_eof() {
                        inner.push(self.parse_pattern()?);
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RParen)?;
                    return Ok(Pattern::Enum { span, path, inner });
                }
                if self.is(&TokenKind::LBrace) {
                    // Struct pattern: Point { x, y }
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.is(&TokenKind::RBrace) && !self.at_eof() {
                        let (_, fname) = self.expect_name()?;
                        let subpat = if self.eat(&TokenKind::Colon) {
                            Some(self.parse_pattern()?)
                        } else {
                            None
                        };
                        fields.push((fname, subpat));
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RBrace)?;
                    return Ok(Pattern::Struct { span, path, fields });
                }
                // Plain identifier
                Ok(Pattern::Ident {
                    span,
                    name: path,
                    mutable: false,
                })
            }
            other => Err(ParseError::new(
                self.current_span(),
                format!("expected pattern, found `{other:?}`"),
            )),
        }
    }
}

// =============================================================================
// §9  EXPRESSIONS  (Pratt / precedence-climbing parser)
// =============================================================================

/// Operator precedence levels (higher = tighter binding).
fn prefix_bp(kind: &TokenKind) -> Option<u8> {
    match kind {
        TokenKind::Minus
        | TokenKind::Bang
        | TokenKind::Tilde
        | TokenKind::Ampersand
        | TokenKind::Star => Some(25),
        TokenKind::KwGrad => Some(28),
        _ => None,
    }
}

/// Returns (left_bp, right_bp) for infix operators.
fn infix_bp(kind: &TokenKind) -> Option<(u8, u8)> {
    match kind {
        // Assignment (lowest, right-assoc)
        TokenKind::Eq
        | TokenKind::PlusEq
        | TokenKind::MinusEq
        | TokenKind::StarEq
        | TokenKind::SlashEq
        | TokenKind::PercentEq
        | TokenKind::AmpEq
        | TokenKind::PipeEq
        | TokenKind::CaretEq
        | TokenKind::MatMulEq => Some((2, 1)),

        TokenKind::KwAs => Some((3, 4)), // cast

        TokenKind::DotDot | TokenKind::DotDotEq => Some((5, 6)), // range

        TokenKind::PipePipe => Some((7, 8)),
        TokenKind::AmpAmp => Some((9, 10)),

        TokenKind::EqEq
        | TokenKind::BangEq
        | TokenKind::Lt
        | TokenKind::Gt
        | TokenKind::LtEq
        | TokenKind::GtEq => Some((11, 12)),

        TokenKind::TensorConcat => Some((13, 14)), // ++
        TokenKind::Pipe => Some((15, 16)),         // bitwise OR
        TokenKind::Caret => Some((17, 18)),        // bitwise XOR
        TokenKind::Ampersand => Some((19, 20)),    // bitwise AND
        TokenKind::LtLt | TokenKind::GtGt => Some((21, 22)), // shifts

        TokenKind::Plus | TokenKind::Minus => Some((23, 24)),

        TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Some((25, 26)),

        // Tensor-specific (tightest)
        TokenKind::MatMul | TokenKind::HadamardMul | TokenKind::HadamardDiv => Some((27, 28)),
        TokenKind::KronProd | TokenKind::OuterProd => Some((27, 28)), // @@ ^*
        TokenKind::FloorDiv => Some((25, 26)),                        // // same prec as / *
        TokenKind::StarStar => Some((29, 28)),                        // right-assoc power

        _ => None,
    }
}

/// Postfix / suffix binding power (method call, index, field).
fn postfix_bp(kind: &TokenKind) -> Option<u8> {
    match kind {
        TokenKind::Dot | TokenKind::LBracket | TokenKind::LParen => Some(31),
        _ => None,
    }
}

impl Parser {
    pub fn parse_expr(&mut self, min_bp: u8) -> ParseResult<Expr> {
        let mut lhs = self.parse_prefix_expr()?;

        loop {
            // Postfix: method call, index, field access
            if let Some(bp) = postfix_bp(&self.peek().kind) {
                if bp <= min_bp {
                    break;
                }
                lhs = self.parse_postfix(lhs)?;
                continue;
            }

            // Infix
            let op_kind = self.peek().kind.clone();
            let Some((l_bp, r_bp)) = infix_bp(&op_kind) else {
                break;
            };
            if l_bp <= min_bp {
                break;
            }

            let op_span = self.advance().span;
            lhs = self.parse_infix(lhs, op_kind, op_span, r_bp)?;
        }

        Ok(lhs)
    }

    fn parse_prefix_expr(&mut self) -> ParseResult<Expr> {
        let span = self.current_span();
        let kind = self.peek().kind.clone();

        // grad expr
        if matches!(kind, TokenKind::KwGrad) {
            self.advance();
            let inner = self.parse_expr(28)?;
            return Ok(Expr::Grad {
                span,
                inner: Box::new(inner),
            });
        }

        // Unary operators
        if let Some(bp) = prefix_bp(&kind) {
            let op = self.parse_unop()?;
            let expr = self.parse_expr(bp)?;
            return Ok(Expr::UnOp {
                span,
                op,
                expr: Box::new(expr),
            });
        }

        self.parse_primary()
    }

    fn parse_unop(&mut self) -> ParseResult<UnOpKind> {
        Ok(match &self.peek().kind {
            TokenKind::Minus => {
                self.advance();
                UnOpKind::Neg
            }
            TokenKind::Bang => {
                self.advance();
                UnOpKind::Not
            }
            TokenKind::Tilde => {
                self.advance();
                UnOpKind::Not
            } // bitwise not
            TokenKind::Ampersand => {
                self.advance();
                if self.eat(&TokenKind::KwMut) {
                    UnOpKind::RefMut
                } else {
                    UnOpKind::Ref
                }
            }
            TokenKind::Star => {
                self.advance();
                UnOpKind::Deref
            }
            other => {
                return Err(ParseError::new(
                    self.current_span(),
                    format!("unknown unary operator `{other:?}`"),
                ))
            }
        })
    }

    fn parse_infix(
        &mut self,
        lhs: Expr,
        op: TokenKind,
        op_span: Span,
        r_bp: u8,
    ) -> ParseResult<Expr> {
        let _span = lhs.span().merge(self.current_span());

        // Assignment
        if let Some(assign_op) = token_to_assign_op(&op) {
            let rhs = self.parse_expr(r_bp)?;
            return Ok(Expr::Assign {
                span: lhs.span().merge(rhs.span()),
                op: assign_op,
                target: Box::new(lhs),
                value: Box::new(rhs),
            });
        }

        // Cast
        if matches!(op, TokenKind::KwAs) {
            let ty = self.parse_type()?;
            return Ok(Expr::Cast {
                span: lhs.span(),
                expr: Box::new(lhs),
                ty,
            });
        }

        // Range
        if matches!(op, TokenKind::DotDot | TokenKind::DotDotEq) {
            let inclusive = matches!(op, TokenKind::DotDotEq);
            let rhs = if !self.is(&TokenKind::RBracket)
                && !self.is(&TokenKind::RBrace)
                && !self.is(&TokenKind::Semicolon)
            {
                Some(Box::new(self.parse_expr(r_bp)?))
            } else {
                None
            };
            return Ok(Expr::Range {
                span: op_span,
                lo: Some(Box::new(lhs)),
                hi: rhs,
                inclusive,
            });
        }

        // Tensor-specific binary operators
        let rhs = self.parse_expr(r_bp)?;
        let full_span = lhs.span().merge(rhs.span());

        match op {
            TokenKind::MatMul => Ok(Expr::MatMul {
                span: full_span,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }),
            TokenKind::HadamardMul => Ok(Expr::HadamardMul {
                span: full_span,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }),
            TokenKind::HadamardDiv => Ok(Expr::HadamardDiv {
                span: full_span,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }),
            TokenKind::TensorConcat => Ok(Expr::TensorConcat {
                span: full_span,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }),
            TokenKind::KronProd => Ok(Expr::KronProd {
                span: full_span,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }),
            TokenKind::OuterProd => Ok(Expr::OuterProd {
                span: full_span,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }),
            TokenKind::StarStar => Ok(Expr::Pow {
                span: full_span,
                base: Box::new(lhs),
                exp: Box::new(rhs),
            }),
            TokenKind::FloorDiv => {
                // Lower to BinOp::FloorDiv
                Ok(Expr::BinOp {
                    span: full_span,
                    op: BinOpKind::FloorDiv,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
            }

            // Normal arithmetic / logical / comparison
            _ => {
                let bin_op = token_to_binop(&op).ok_or_else(|| {
                    ParseError::new(op_span, format!("unknown binary operator `{op:?}`"))
                })?;
                Ok(Expr::BinOp {
                    span: full_span,
                    op: bin_op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
            }
        }
    }

    fn parse_postfix(&mut self, lhs: Expr) -> ParseResult<Expr> {
        match self.peek().kind.clone() {
            // Field or method access
            TokenKind::Dot => {
                let _dot = self.advance();
                let (field_span, field) = self.expect_name()?;
                let full = lhs.span().merge(field_span);

                // Method call: expr.method(args)
                if self.is(&TokenKind::LParen) {
                    let args = self.parse_call_args()?;
                    return Ok(Expr::MethodCall {
                        span: full,
                        receiver: Box::new(lhs),
                        method: field,
                        args,
                    });
                }

                Ok(Expr::Field {
                    span: full,
                    object: Box::new(lhs),
                    field,
                })
            }

            // Index: expr[i] or expr[i, j]
            TokenKind::LBracket => {
                let _lb = self.advance();
                let mut indices = Vec::new();
                while !self.is(&TokenKind::RBracket) && !self.at_eof() {
                    indices.push(self.parse_expr(0)?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                let rb = self.expect(&TokenKind::RBracket)?;
                Ok(Expr::Index {
                    span: lhs.span().merge(rb),
                    object: Box::new(lhs),
                    indices,
                })
            }

            // Call: expr(args)
            TokenKind::LParen => {
                let args = self.parse_call_args()?;
                let span = lhs.span();
                Ok(Expr::Call {
                    span,
                    func: Box::new(lhs),
                    args,
                    named: vec![],
                })
            }

            _ => Ok(lhs),
        }
    }

    fn parse_call_args(&mut self) -> ParseResult<Vec<Expr>> {
        self.expect(&TokenKind::LParen)?;
        let mut args = Vec::new();
        while !self.is(&TokenKind::RParen) && !self.at_eof() {
            // Skip named args (name: val) — consume just the value for now.
            if matches!(&self.peek().kind, TokenKind::Ident(_))
                && self.peek2().kind == TokenKind::Colon
            {
                self.advance(); // skip name
                self.advance(); // skip ':'
            }
            args.push(self.parse_expr(0)?);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok(args)
    }

    // ── Primary expressions ───────────────────────────────────────────────────

    fn parse_primary(&mut self) -> ParseResult<Expr> {
        let span = self.current_span();
        match self.peek().kind.clone() {
            // Literals
            TokenKind::IntLit { value, .. } => {
                self.advance();
                Ok(Expr::IntLit { span, value })
            }
            TokenKind::FloatLit { value } => {
                self.advance();
                Ok(Expr::FloatLit { span, value })
            }
            TokenKind::BoolLit { value } => {
                self.advance();
                Ok(Expr::BoolLit { span, value })
            }
            TokenKind::StringLit { value } => {
                self.advance();
                Ok(Expr::StrLit { span, value })
            }

            // Block expression
            TokenKind::LBrace => {
                let b = self.parse_block()?;
                Ok(Expr::Block(Box::new(b)))
            }

            // Tuple / grouped expression
            TokenKind::LParen => {
                self.advance();
                if self.eat(&TokenKind::RParen) {
                    return Ok(Expr::Tuple {
                        span,
                        elems: vec![],
                    });
                }
                let first = self.parse_expr(0)?;
                if self.is(&TokenKind::RParen) {
                    self.advance();
                    return Ok(first); // parenthesised
                }
                let mut elems = vec![first];
                while self.eat(&TokenKind::Comma) {
                    if self.is(&TokenKind::RParen) {
                        break;
                    }
                    elems.push(self.parse_expr(0)?);
                }
                self.expect(&TokenKind::RParen)?;
                Ok(Expr::Tuple { span, elems })
            }

            // Array literal: [1, 2, 3]
            TokenKind::LBracket => {
                self.advance();
                let mut elems = Vec::new();
                while !self.is(&TokenKind::RBracket) && !self.at_eof() {
                    elems.push(self.parse_expr(0)?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RBracket)?;
                Ok(Expr::ArrayLit { span, elems })
            }

            // if expr { … } else { … }  — expression form
            TokenKind::KwIf => {
                self.advance();
                let cond = self.parse_expr(0)?;
                let then = self.parse_block()?;
                let else_ = if self.eat(&TokenKind::KwElse) {
                    Some(Box::new(self.parse_block()?))
                } else {
                    None
                };
                Ok(Expr::IfExpr {
                    span,
                    cond: Box::new(cond),
                    then: Box::new(then),
                    else_,
                })
            }

            // Closure: |x, y| expr
            TokenKind::Pipe => {
                self.advance();
                let mut params = Vec::new();
                while !self.is(&TokenKind::Pipe) && !self.at_eof() {
                    params.push(self.parse_param()?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::Pipe)?;
                let ret_ty = if self.eat(&TokenKind::Arrow) {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                let body = self.parse_expr(0)?;
                Ok(Expr::Closure {
                    span,
                    params,
                    ret_ty,
                    body: Box::new(body),
                })
            }

            // Vec constructor: vec3(x, y, z)
            TokenKind::KwVec2 | TokenKind::KwVec3 | TokenKind::KwVec4 => {
                let size = match &self.peek().kind {
                    TokenKind::KwVec2 => VecSize::N2,
                    TokenKind::KwVec3 => VecSize::N3,
                    _ => VecSize::N4,
                };
                self.advance();
                let elems = self.parse_call_args()?;
                Ok(Expr::VecCtor { span, size, elems })
            }

            // Identifier, path, or struct literal
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();

                // Path: A::B::C
                if self.is(&TokenKind::ColonColon) {
                    let mut segs = vec![name];
                    while self.eat(&TokenKind::ColonColon) {
                        segs.push(self.expect_name()?.1);
                    }
                    return Ok(Expr::Path {
                        span,
                        segments: segs,
                    });
                }

                // Struct literal: Foo { x: 1, y: 2 }
                if self.is(&TokenKind::LBrace)
                    && name
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                {
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.is(&TokenKind::RBrace) && !self.at_eof() {
                        let (_, fname) = self.expect_name()?;
                        self.expect(&TokenKind::Colon)?;
                        let fval = self.parse_expr(0)?;
                        fields.push((fname, fval));
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RBrace)?;
                    return Ok(Expr::StructLit { span, name, fields });
                }

                Ok(Expr::Ident { span, name })
            }

            // Keywords usable as identifiers in expression position
            TokenKind::KwWorld | TokenKind::KwSelf => {
                let name = match &self.peek().kind {
                    TokenKind::KwWorld => "world",
                    TokenKind::KwSelf => "self",
                    _ => unreachable!(),
                }
                .to_owned();
                self.advance();
                Ok(Expr::Ident { span, name })
            }

            other => Err(ParseError::new(
                span,
                format!("unexpected token `{other:?}` in expression"),
            )),
        }
    }
}

fn token_to_binop(kind: &TokenKind) -> Option<BinOpKind> {
    Some(match kind {
        TokenKind::Plus => BinOpKind::Add,
        TokenKind::Minus => BinOpKind::Sub,
        TokenKind::Star => BinOpKind::Mul,
        TokenKind::Slash => BinOpKind::Div,
        TokenKind::Percent => BinOpKind::Rem,
        TokenKind::EqEq => BinOpKind::Eq,
        TokenKind::BangEq => BinOpKind::Ne,
        TokenKind::Lt => BinOpKind::Lt,
        TokenKind::LtEq => BinOpKind::Le,
        TokenKind::Gt => BinOpKind::Gt,
        TokenKind::GtEq => BinOpKind::Ge,
        TokenKind::AmpAmp => BinOpKind::And,
        TokenKind::PipePipe => BinOpKind::Or,
        TokenKind::Ampersand => BinOpKind::BitAnd,
        TokenKind::Pipe => BinOpKind::BitOr,
        TokenKind::Caret => BinOpKind::BitXor,
        TokenKind::LtLt => BinOpKind::Shl,
        TokenKind::GtGt => BinOpKind::Shr,
        _ => return None,
    })
}

fn token_to_assign_op(kind: &TokenKind) -> Option<AssignOpKind> {
    Some(match kind {
        TokenKind::Eq => AssignOpKind::Assign,
        TokenKind::PlusEq => AssignOpKind::AddAssign,
        TokenKind::MinusEq => AssignOpKind::SubAssign,
        TokenKind::StarEq => AssignOpKind::MulAssign,
        TokenKind::SlashEq => AssignOpKind::DivAssign,
        TokenKind::PercentEq => AssignOpKind::RemAssign,
        TokenKind::AmpEq => AssignOpKind::BitAndAssign,
        TokenKind::PipeEq => AssignOpKind::BitOrAssign,
        TokenKind::CaretEq => AssignOpKind::BitXorAssign,
        TokenKind::MatMulEq => AssignOpKind::MatMulAssign,
        _ => return None,
    })
}

// =============================================================================
// §10  AGENT DECLARATION
// =============================================================================

impl Parser {
    fn parse_agent(&mut self, attrs: Vec<Attribute>) -> ParseResult<AgentDecl> {
        let start = self.expect(&TokenKind::KwAgent)?;
        let (_, name) = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;

        let mut perceptions = Vec::new();
        let mut memory = None;
        let mut learning = None;
        let mut behaviors = Vec::new();
        let mut goals = Vec::new();
        let mut fields = Vec::new();
        let mut architecture = AgentArchitecture::Utility;

        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            match self.peek().kind.clone() {
                TokenKind::KwPerception => {
                    perceptions.push(self.parse_perception()?);
                }
                TokenKind::KwMemory => {
                    memory = Some(self.parse_memory()?);
                }
                TokenKind::KwLearning => {
                    learning = Some(self.parse_learning()?);
                }
                TokenKind::KwBehavior => {
                    behaviors.push(self.parse_behavior()?);
                }
                TokenKind::KwGoal => {
                    goals.push(self.parse_goal()?);
                }
                TokenKind::Ident(s) if s == "architecture" => {
                    self.advance();
                    self.eat(&TokenKind::Colon);
                    let (_, arch_name) = self.expect_ident()?;
                    architecture = match arch_name.as_str() {
                        "fsm" => AgentArchitecture::Fsm,
                        "behavior_tree" => AgentArchitecture::BehaviorTree,
                        "utility" => AgentArchitecture::Utility,
                        "goap" => AgentArchitecture::Goap,
                        "learned" => AgentArchitecture::Learned,
                        _ => AgentArchitecture::Utility,
                    };
                }
                // Field declarations inside agent body
                _ => {
                    let field_span = self.current_span();
                    let fattrs = self.parse_attrs();
                    match self.expect_name() {
                        Ok((_, fname)) => {
                            if self.eat(&TokenKind::Colon) {
                                if let Ok(ftype) = self.parse_type() {
                                    fields.push(StructField {
                                        span: field_span,
                                        name: fname,
                                        ty: ftype,
                                        attrs: fattrs,
                                    });
                                }
                            }
                        }
                        Err(e) => {
                            self.recover(e);
                            continue;
                        }
                    }
                }
            }
            self.eat(&TokenKind::Semicolon);
            self.eat(&TokenKind::Comma);
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(AgentDecl {
            span: start.merge(end),
            attrs,
            name,
            architecture,
            perceptions,
            memory,
            learning,
            behaviors,
            goals,
            fields,
        })
    }

    fn parse_perception(&mut self) -> ParseResult<PerceptionSpec> {
        let span = self.expect(&TokenKind::KwPerception)?;
        let (_, kind_name) = self.expect_name()?;
        let kind = match kind_name.as_str() {
            "vision" => PerceptionKind::Vision,
            "hearing" => PerceptionKind::Hearing,
            "omniscient" => PerceptionKind::Omniscient,
            other => PerceptionKind::Custom(other.into()),
        };
        let range = match &self.peek().kind.clone() {
            TokenKind::IntLit { value, .. } => {
                let v = *value as f64;
                self.advance();
                Some(v)
            }
            TokenKind::FloatLit { value } => {
                let v = *value;
                self.advance();
                Some(v)
            }
            _ => None,
        };
        Ok(PerceptionSpec {
            span,
            kind,
            range,
            fov: None,
            tag: None,
        })
    }

    fn parse_memory(&mut self) -> ParseResult<MemorySpec> {
        let span = self.expect(&TokenKind::KwMemory)?;
        let (_, kind_name) = self.expect_name()?;
        let kind = match kind_name.as_str() {
            "episodic" => MemoryKind::Episodic,
            "semantic" => MemoryKind::Semantic,
            "working" => MemoryKind::Working,
            "external" => MemoryKind::External(Box::new(self.parse_type()?)),
            _other => MemoryKind::Episodic, // fallback
        };
        let capacity = match &self.peek().kind.clone() {
            TokenKind::IntLit { value, .. } => {
                let v = *value as u64;
                self.advance();
                // Check for suffix 's' for seconds  (treated as ident)
                if matches!(&self.peek().kind, TokenKind::Ident(s) if s == "s") {
                    self.advance();
                    Some(MemoryCapacity::Duration { seconds: v as f64 })
                } else {
                    Some(MemoryCapacity::Slots(v))
                }
            }
            _ => None,
        };
        Ok(MemorySpec {
            span,
            kind,
            capacity,
        })
    }

    fn parse_learning(&mut self) -> ParseResult<LearningSpec> {
        let span = self.expect(&TokenKind::KwLearning)?;
        let (_, kind_name) = self.expect_name()?;
        let kind = match kind_name.as_str() {
            "reinforcement" => LearningKind::Reinforcement,
            "imitation" => LearningKind::Imitation,
            "evolutionary" => LearningKind::Evolutionary,
            "none" => LearningKind::None,
            other => LearningKind::Custom(other.into()),
        };
        let mut learning_rate = None;
        let mut gamma = None;
        let mut policy_model = None;
        // Optional: learning_rate: 0.001, gamma: 0.99, model: PolicyNet
        while self.eat(&TokenKind::Comma) || matches!(self.peek().kind, TokenKind::Ident(_)) {
            if !matches!(&self.peek().kind, TokenKind::Ident(_)) {
                break;
            }
            let (_, key) = self.expect_ident()?;
            self.eat(&TokenKind::Colon);
            match key.as_str() {
                "learning_rate" | "lr" => {
                    learning_rate = Some(
                        self.parse_expr(0)
                            .ok()
                            .and_then(|e| expr_to_f64(&e))
                            .unwrap_or(3e-4),
                    );
                }
                "gamma" => {
                    gamma = Some(
                        self.parse_expr(0)
                            .ok()
                            .and_then(|e| expr_to_f64(&e))
                            .unwrap_or(0.99),
                    );
                }
                "model" | "policy_model" => {
                    policy_model = Some(self.expect_ident()?.1);
                }
                _ => {}
            }
        }
        Ok(LearningSpec {
            span,
            kind,
            learning_rate,
            gamma,
            policy_model,
        })
    }

    fn parse_behavior(&mut self) -> ParseResult<BehaviorRule> {
        let span = self.expect(&TokenKind::KwBehavior)?;
        let (_, name) = self.expect_ident()?;
        // (priority: 100)
        let mut priority = 0u32;
        if self.is(&TokenKind::LParen) {
            self.advance();
            while !self.is(&TokenKind::RParen) && !self.at_eof() {
                let (_, key) = self.expect_name()?;
                self.eat(&TokenKind::Colon);
                if key == "priority" {
                    if let TokenKind::IntLit { value, .. } = self.peek().kind.clone() {
                        priority = value as u32;
                        self.advance();
                    }
                } else {
                    self.parse_expr(0)?; // consume value
                }
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RParen)?;
        }
        let params = if self.is(&TokenKind::LParen) {
            self.parse_params()?
        } else {
            vec![]
        };
        self.eat(&TokenKind::Colon);
        let body = self.parse_block()?;
        Ok(BehaviorRule {
            span,
            name,
            priority: BehaviorPriority(priority),
            params,
            body,
        })
    }

    fn parse_goal(&mut self) -> ParseResult<GoalDecl> {
        let span = self.expect(&TokenKind::KwGoal)?;
        let (_, name) = self.expect_ident()?;
        self.eat(&TokenKind::Colon);
        // utility = <expr>
        let utility = if matches!(&self.peek().kind, TokenKind::Ident(s) if s == "utility") {
            self.advance();
            self.expect(&TokenKind::Eq)?;
            self.parse_expr(0)?
        } else {
            self.parse_expr(0)?
        };
        Ok(GoalDecl {
            span,
            name,
            utility,
        })
    }
}

fn _dummy() -> TokenKind {
    TokenKind::Ident(String::new())
}

fn expr_to_f64(e: &Expr) -> Option<f64> {
    match e {
        Expr::FloatLit { value, .. } => Some(*value),
        Expr::IntLit { value, .. } => Some(*value as f64),
        _ => None,
    }
}

// =============================================================================
// §11  MODEL DECLARATION
// =============================================================================

impl Parser {
    fn parse_model(&mut self, attrs: Vec<Attribute>) -> ParseResult<ModelDecl> {
        let start = self.expect(&TokenKind::KwModel)?;
        let (_, name) = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;
        let mut layers = Vec::new();

        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            match self.parse_model_layer() {
                Ok(l) => layers.push(l),
                Err(e) => self.recover(e),
            }
            self.eat(&TokenKind::Semicolon);
            self.eat(&TokenKind::Comma);
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(ModelDecl {
            span: start.merge(end),
            attrs,
            name,
            layers,
            device: ModelDevice::Auto,
            optimizer: None,
        })
    }

    fn parse_model_layer(&mut self) -> ParseResult<ModelLayer> {
        let span = self.current_span();
        match self.peek().kind.clone() {
            TokenKind::KwLayer => {
                // Optional `layer` prefix: `layer dense 128 relu`
                self.advance();
                return self.parse_model_layer();
            }
            TokenKind::KwInput => {
                self.advance();
                let size = self.expect_u64()?;
                Ok(ModelLayer::Input { span, size })
            }
            TokenKind::KwOutput => {
                self.advance();
                let units = self.expect_u64()?;
                let activation = self.try_parse_activation();
                Ok(ModelLayer::Output {
                    span,
                    units,
                    activation,
                })
            }
            TokenKind::KwDense => {
                self.advance();
                let units = self.expect_u64()?;
                let activation = self.try_parse_activation();
                Ok(ModelLayer::Dense {
                    span,
                    units,
                    activation,
                    bias: true,
                })
            }
            TokenKind::KwConv => {
                self.advance();
                let filters = self.expect_u64()?;
                // kernel: NxM  (e.g. 3x3) — parsed as N * x * M tokens or an ident "3x3"
                let (kh, kw) = self.parse_kernel_size()?;
                let activation = self.try_parse_activation();
                Ok(ModelLayer::Conv2d {
                    span,
                    filters,
                    kernel_h: kh,
                    kernel_w: kw,
                    stride: 1,
                    padding: Padding::Same,
                    activation,
                })
            }
            TokenKind::KwPool => {
                self.advance();
                let (ph, pw) = self.parse_kernel_size()?;
                let op = self.parse_pool_op();
                Ok(ModelLayer::Pool {
                    span,
                    size_h: ph,
                    size_w: pw,
                    op,
                })
            }
            TokenKind::KwDropout => {
                self.advance();
                let rate = match &self.peek().kind.clone() {
                    TokenKind::FloatLit { value } => {
                        let v = *value;
                        self.advance();
                        v
                    }
                    TokenKind::IntLit { value, .. } => {
                        let v = *value as f64;
                        self.advance();
                        v
                    }
                    _ => 0.0,
                };
                Ok(ModelLayer::Dropout { span, rate })
            }
            TokenKind::KwNorm => {
                self.advance();
                let kind = match self.peek().kind.clone() {
                    TokenKind::Ident(s) => {
                        self.advance();
                        match s.as_str() {
                            "batch" => NormKind::Batch,
                            "layer" => NormKind::Layer,
                            "rms" => NormKind::Rms,
                            "group" => NormKind::Group,
                            _ => NormKind::Layer,
                        }
                    }
                    _ => NormKind::Layer,
                };
                Ok(ModelLayer::Norm { span, kind })
            }
            TokenKind::KwAttention => {
                self.advance();
                let num_heads = self.expect_u64()?;
                let head_dim = self.expect_u64()?;
                Ok(ModelLayer::Attention {
                    span,
                    num_heads,
                    head_dim,
                })
            }
            TokenKind::KwEmbed => {
                self.advance();
                let vocab_size = self.expect_u64()?;
                let embed_dim = self.expect_u64()?;
                Ok(ModelLayer::Embed {
                    span,
                    vocab_size,
                    embed_dim,
                })
            }
            TokenKind::KwRecurrent => {
                self.advance();
                let units = self.expect_u64()?;
                let cell = match self.peek().kind.clone() {
                    TokenKind::Ident(s) => {
                        self.advance();
                        match s.as_str() {
                            "lstm" => RecurrentCell::Lstm,
                            "gru" => RecurrentCell::Gru,
                            _ => RecurrentCell::SimpleRnn,
                        }
                    }
                    _ => RecurrentCell::Lstm,
                };
                Ok(ModelLayer::Recurrent {
                    span,
                    units,
                    cell,
                    bidirect: false,
                })
            }
            // Sub-model reference by name
            TokenKind::Ident(name) => {
                let name = name.clone();
                let nspan = self.current_span();
                self.advance();
                // `residual { … }` sub-block
                if name == "residual" && self.is(&TokenKind::LBrace) {
                    self.expect(&TokenKind::LBrace)?;
                    let mut inner = Vec::new();
                    while !self.is(&TokenKind::RBrace) && !self.at_eof() {
                        match self.parse_model_layer() {
                            Ok(l) => inner.push(l),
                            Err(e) => self.recover(e),
                        }
                        self.eat(&TokenKind::Semicolon);
                        self.eat(&TokenKind::Comma);
                    }
                    self.expect(&TokenKind::RBrace)?;
                    return Ok(ModelLayer::Residual {
                        span: nspan,
                        layers: inner,
                    });
                }
                // `flatten` — collapse spatial dims
                if name == "flatten" {
                    return Ok(ModelLayer::Flatten { span: nspan });
                }
                Ok(ModelLayer::SubModel { span, name })
            }
            other => Err(ParseError::new(
                span,
                format!("expected model layer declaration, found `{other:?}`"),
            )),
        }
    }

    fn expect_u64(&mut self) -> ParseResult<u64> {
        match self.peek().kind.clone() {
            TokenKind::IntLit { value, .. } => {
                self.advance();
                Ok(value as u64)
            }
            other => Err(ParseError::new(
                self.current_span(),
                format!("expected integer, found `{other:?}`"),
            )),
        }
    }

    fn parse_kernel_size(&mut self) -> ParseResult<(u64, u64)> {
        // Accepts "3x3" as ident or separate "3 3"
        match self.peek().kind.clone() {
            TokenKind::Ident(s) if s.contains('x') => {
                self.advance();
                let parts: Vec<&str> = s.splitn(2, 'x').collect();
                let h = parts[0].parse::<u64>().unwrap_or(1);
                let w = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(h);
                Ok((h, w))
            }
            TokenKind::IntLit { value: h, .. } => {
                let h = h as u64;
                self.advance();
                let w = if matches!(&self.peek().kind, TokenKind::IntLit { .. }) {
                    self.expect_u64()?
                } else {
                    h
                };
                Ok((h, w))
            }
            _ => Ok((1, 1)),
        }
    }

    fn try_parse_activation(&mut self) -> Activation {
        match &self.peek().kind {
            TokenKind::KwRelu => {
                self.advance();
                Activation::Relu
            }
            TokenKind::KwLeakyRelu => {
                self.advance();
                Activation::LeakyRelu
            }
            TokenKind::KwSigmoid => {
                self.advance();
                Activation::Sigmoid
            }
            TokenKind::KwTanh => {
                self.advance();
                Activation::Tanh
            }
            TokenKind::KwGelu => {
                self.advance();
                Activation::Gelu
            }
            TokenKind::KwSilu => {
                self.advance();
                Activation::Silu
            }
            TokenKind::KwSoftmax => {
                self.advance();
                Activation::Softmax
            }
            TokenKind::KwLinear => {
                self.advance();
                Activation::Linear
            }
            TokenKind::Ident(s) => {
                let name = s.clone();
                self.advance();
                Activation::Custom(name)
            }
            _ => Activation::Linear,
        }
    }

    fn parse_pool_op(&mut self) -> PoolOp {
        match &self.peek().kind {
            TokenKind::Ident(s) => {
                let op = match s.as_str() {
                    "max" => PoolOp::Max,
                    "avg" | "average" => PoolOp::Average,
                    "global_max" => PoolOp::GlobalMax,
                    "global_avg" | "global_average" => PoolOp::GlobalAverage,
                    _ => PoolOp::Max,
                };
                self.advance();
                op
            }
            _ => PoolOp::Max,
        }
    }
}

// =============================================================================
// §12  TRAIN DECLARATION
// =============================================================================

impl Parser {
    fn parse_train(&mut self, attrs: Vec<Attribute>) -> ParseResult<TrainDecl> {
        let start = self.expect(&TokenKind::KwTrain)?;
        let (_, agent) = self.expect_ident()?;
        // `in WorldName`
        self.expect(&TokenKind::KwIn)?;
        let (_, world) = self.expect_ident()?;

        self.expect(&TokenKind::LBrace)?;
        let mut signals = Vec::new();
        let mut episode = None;
        let mut model = None;
        let mut optimizer = None;
        let mut hyper = Vec::new();
        let mut algorithm = None;
        let mut value_model = None;

        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            match self.peek().kind.clone() {
                TokenKind::KwReward | TokenKind::KwPenalty => {
                    let is_reward = matches!(self.peek().kind, TokenKind::KwReward);
                    let sig_span = self.advance().span;
                    let (_, name) = self.expect_name()?;
                    let weight = if matches!(
                        &self.peek().kind,
                        TokenKind::FloatLit { .. } | TokenKind::IntLit { .. }
                    ) {
                        match self.peek().kind.clone() {
                            TokenKind::FloatLit { value } => {
                                self.advance();
                                value
                            }
                            TokenKind::IntLit { value, .. } => {
                                self.advance();
                                value as f64
                            }
                            _ => 1.0,
                        }
                    } else {
                        1.0
                    };
                    signals.push(SignalSpec {
                        span: sig_span,
                        is_reward,
                        name,
                        weight,
                        expr: None,
                    });
                }
                TokenKind::KwEpisode => {
                    episode = Some(self.parse_episode_spec()?);
                }
                TokenKind::KwModel => {
                    self.advance();
                    if matches!(&self.peek().kind, TokenKind::Ident(s) if s == "value") {
                        self.advance(); // consume "value"
                        value_model = Some(self.expect_ident()?.1);
                    } else {
                        model = Some(self.expect_ident()?.1);
                    }
                }
                TokenKind::Ident(key) if key == "optimizer" => {
                    self.advance();
                    self.eat(&TokenKind::Colon);
                    optimizer = Some(self.parse_optimizer_spec(start)?);
                }
                // `algorithm: ppo | dqn | sac | reinforce`
                TokenKind::Ident(key) if key == "algorithm" => {
                    self.advance();
                    self.eat(&TokenKind::Colon);
                    let (_, alg_name) = self.expect_name()?;
                    algorithm = Some(alg_name);
                }
                // hyper-parameter: key = value
                TokenKind::Ident(key) => {
                    let key = key.clone();
                    self.advance();
                    self.eat(&TokenKind::Eq);
                    let val = self.parse_expr(0)?;
                    hyper.push((key, val));
                }
                _ => {
                    self.advance();
                } // skip unknown
            }
            self.eat(&TokenKind::Semicolon);
            self.eat(&TokenKind::Comma);
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(TrainDecl {
            span: start.merge(end),
            attrs,
            agent,
            world,
            signals,
            episode,
            model,
            optimizer,
            hyper,
            algorithm,
            value_model,
        })
    }

    fn parse_episode_spec(&mut self) -> ParseResult<EpisodeSpec> {
        let span = self.expect(&TokenKind::KwEpisode)?;
        self.expect(&TokenKind::LBrace)?;
        let mut max_steps = None;
        let mut max_seconds = None;
        let mut done_condition = None;
        let mut num_envs = None;

        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            let (_, key) = self
                .expect_name()
                .unwrap_or((Span::dummy(), "".to_string()));
            self.eat(&TokenKind::Colon);
            match key.as_str() {
                "max_steps" => {
                    max_steps = Some(self.expect_u64()?);
                }
                "max_seconds" => {
                    max_seconds = Some(match self.peek().kind.clone() {
                        TokenKind::FloatLit { value } => {
                            self.advance();
                            value
                        }
                        TokenKind::IntLit { value, .. } => {
                            self.advance();
                            value as f64
                        }
                        _ => 0.0,
                    });
                }
                "done" | "done_condition" => {
                    done_condition = Some(self.parse_expr(0)?);
                }
                "num_envs" => {
                    num_envs = Some(self.expect_u64()?);
                }
                _ => {
                    self.parse_expr(0).ok();
                }
            }
            self.eat(&TokenKind::Comma);
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(EpisodeSpec {
            span,
            max_steps,
            max_seconds,
            done_condition,
            num_envs,
        })
    }

    fn parse_optimizer_spec(&mut self, span: Span) -> ParseResult<OptimizerSpec> {
        let (_, kind_name) = self.expect_name()?;
        let kind = match kind_name.as_str() {
            "adam" => OptimizerKind::Adam,
            "sgd" => OptimizerKind::Sgd,
            "rmsprop" => OptimizerKind::Rmsprop,
            "adagrad" => OptimizerKind::Adagrad,
            "adamw" => OptimizerKind::AdamW,
            "lion" => OptimizerKind::Lion,
            "sophia" => OptimizerKind::Sophia,
            "prodigy" => OptimizerKind::Prodigy,
            _ => OptimizerKind::Adam,
        };
        let mut lr = 3e-4;
        let mut extra = Vec::new();
        if self.is(&TokenKind::LBrace) {
            self.advance();
            while !self.is(&TokenKind::RBrace) && !self.at_eof() {
                let (_, k) = self
                    .expect_name()
                    .unwrap_or((Span::dummy(), "".to_string()));
                self.eat(&TokenKind::Colon);
                let v = match self.peek().kind.clone() {
                    TokenKind::FloatLit { value } => {
                        self.advance();
                        value
                    }
                    TokenKind::IntLit { value, .. } => {
                        self.advance();
                        value as f64
                    }
                    _ => 0.0,
                };
                if k == "lr" || k == "learning_rate" {
                    lr = v;
                } else {
                    extra.push((k, v));
                }
                self.eat(&TokenKind::Comma);
            }
            self.expect(&TokenKind::RBrace)?;
        }
        Ok(OptimizerSpec {
            span,
            kind,
            learning_rate: lr,
            extra,
        })
    }
}

// =============================================================================
// §13  SHADER / SCENE / PREFAB / PHYSICS / LOSS  (new top-level items)
// =============================================================================

impl Parser {
    // ── shader Name { vertex { … } fragment { … } compute { … } } ────────────
    fn parse_shader(&mut self, attrs: Vec<Attribute>) -> ParseResult<ShaderDecl> {
        let start = self.expect(&TokenKind::KwShader)?;
        let (_, name) = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;

        let mut bindings = Vec::new();
        let mut vertex_body = None;
        let mut fragment_body = None;
        let mut compute_body = None;

        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            match self.peek().kind.clone() {
                // Uniform / buffer / sampler / texture bindings
                TokenKind::KwUniform
                | TokenKind::KwBuffer
                | TokenKind::KwSampler
                | TokenKind::KwTexture => {
                    let bkind_tok = self.advance().kind.clone();
                    let (bspan, bname) = self.expect_ident()?;
                    self.eat(&TokenKind::Colon);
                    let bty = self.parse_type()?;
                    // Optional binding index: @binding(0)
                    let binding_idx = if let TokenKind::AtKernel = self.peek().kind.clone() {
                        None
                    } else if self.is(&TokenKind::Hash) {
                        self.advance();
                        self.expect_u64().ok()
                    } else {
                        None
                    };
                    let bkind = match bkind_tok {
                        TokenKind::KwUniform => ShaderBindingKind::Uniform,
                        TokenKind::KwBuffer => ShaderBindingKind::Storage,
                        TokenKind::KwSampler => ShaderBindingKind::Sampler,
                        TokenKind::KwTexture => ShaderBindingKind::Texture,
                        _ => ShaderBindingKind::Uniform,
                    };
                    bindings.push(ShaderBinding {
                        span: bspan,
                        name: bname,
                        ty: bty,
                        kind: bkind,
                        binding_idx,
                    });
                    self.eat(&TokenKind::Semicolon);
                    self.eat(&TokenKind::Comma);
                }
                TokenKind::KwVertex => {
                    self.advance();
                    vertex_body = Some(self.parse_block()?);
                }
                TokenKind::KwFragment => {
                    self.advance();
                    fragment_body = Some(self.parse_block()?);
                }
                TokenKind::KwCompute => {
                    self.advance();
                    compute_body = Some(self.parse_block()?);
                }
                _ => {
                    self.advance();
                }
            }
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(ShaderDecl {
            span: start.merge(end),
            attrs,
            name,
            bindings,
            vertex: vertex_body.map(Box::new),
            fragment: fragment_body.map(Box::new),
            compute: compute_body.map(Box::new),
        })
    }

    // ── scene Name { instantiate Prefab, … } ─────────────────────────────────
    fn parse_scene(&mut self, attrs: Vec<Attribute>) -> ParseResult<SceneDecl> {
        let start = self.expect(&TokenKind::KwScene)?;
        let (_, name) = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;
        let mut instances = Vec::new();

        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            let ispan = self.current_span();
            // `PrefabName { field: val, … }` or just `PrefabName`
            let (_, prefab_name) = self.expect_ident()?;
            let overrides = if self.is(&TokenKind::LBrace) {
                self.parse_struct_fields_values()?
            } else {
                vec![]
            };
            instances.push(SceneInstance {
                span: ispan,
                prefab: prefab_name,
                overrides,
            });
            self.eat(&TokenKind::Comma);
            self.eat(&TokenKind::Semicolon);
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(SceneDecl {
            span: start.merge(end),
            attrs,
            name,
            instances,
        })
    }

    // ── prefab Name { component Component, … } ────────────────────────────────
    fn parse_prefab(&mut self, attrs: Vec<Attribute>) -> ParseResult<PrefabDecl> {
        let start = self.expect(&TokenKind::KwPrefab)?;
        let (_, name) = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;
        let mut components = Vec::new();

        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            let cspan = self.current_span();
            let (_, cname) = self.expect_ident()?;
            let fields = if self.is(&TokenKind::LBrace) {
                self.parse_struct_fields_values()?
            } else {
                vec![]
            };
            components.push(PrefabComponent {
                span: cspan,
                name: cname,
                fields,
            });
            self.eat(&TokenKind::Comma);
            self.eat(&TokenKind::Semicolon);
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(PrefabDecl {
            span: start.merge(end),
            attrs,
            name,
            components,
        })
    }

    // ── physics { gravity: vec3(0,-9.8,0), iterations: 10, … } ──────────────
    fn parse_physics_config(&mut self, attrs: Vec<Attribute>) -> ParseResult<PhysicsConfigDecl> {
        let start = self.expect(&TokenKind::KwPhysics)?;
        self.expect(&TokenKind::LBrace)?;
        let mut gravity = None;
        let mut iterations = None;
        let mut substeps = None;
        let mut collision_layers = Vec::new();

        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            let (_, key) = self.expect_name().unwrap_or((Span::dummy(), String::new()));
            self.eat(&TokenKind::Colon);
            match key.as_str() {
                "gravity" => {
                    gravity = Some(self.parse_expr(0)?);
                }
                "iterations" => {
                    iterations = self.expect_u64().ok();
                }
                "substeps" => {
                    substeps = self.expect_u64().ok();
                }
                "layer" => {
                    let (_, lname) = self
                        .expect_ident()
                        .unwrap_or((Span::dummy(), String::new()));
                    collision_layers.push(lname);
                }
                _ => {
                    let _ = self.parse_expr(0);
                }
            }
            self.eat(&TokenKind::Comma);
            self.eat(&TokenKind::Semicolon);
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(PhysicsConfigDecl {
            span: start.merge(end),
            attrs,
            gravity,
            iterations,
            substeps,
            collision_layers,
        })
    }

    // ── loss LossName { fn forward(pred, target) -> f32 { … } } ─────────────
    fn parse_loss(&mut self, attrs: Vec<Attribute>) -> ParseResult<LossDecl> {
        let start = self.expect(&TokenKind::KwLoss)?;
        let (_, name) = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;
        let mut methods = Vec::new();

        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            match self.peek().kind.clone() {
                TokenKind::KwFn | TokenKind::KwAsync => {
                    let fn_attrs = self.parse_attrs();
                    match self.parse_fn(fn_attrs) {
                        Ok(f) => methods.push(f),
                        Err(e) => self.recover(e),
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(LossDecl {
            span: start.merge(end),
            attrs,
            name,
            methods,
        })
    }

    // ── helper: parse `{ key: expr, … }` field-value pairs ───────────────────
    fn parse_struct_fields_values(&mut self) -> ParseResult<Vec<(String, Expr)>> {
        self.expect(&TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.is(&TokenKind::RBrace) && !self.at_eof() {
            let (_, fname) = self.expect_name()?;
            self.expect(&TokenKind::Colon)?;
            let fval = self.parse_expr(0)?;
            fields.push((fname, fval));
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(fields)
    }
}

// =============================================================================
// §14  TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse(src: &str) -> (Program, Vec<ParseError>) {
        let mut lexer = Lexer::new(src);
        let (tokens, lex_errors) = lexer.tokenize();
        assert!(lex_errors.is_empty(), "lex errors: {lex_errors:?}");
        let mut parser = Parser::new(tokens);
        let prog = parser.parse_program();
        let errs = parser.errors.clone();
        (prog, errs)
    }

    fn ok(src: &str) -> Program {
        let (prog, errs) = parse(src);
        assert!(errs.is_empty(), "unexpected parse errors: {errs:#?}");
        prog
    }

    fn has_error(src: &str) -> bool {
        let (_, errs) = parse(src);
        !errs.is_empty()
    }

    // ── §4  Functions ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_empty_fn() {
        let prog = ok("fn foo() {}");
        assert_eq!(prog.items.len(), 1);
        assert!(matches!(&prog.items[0], Item::Fn(f) if f.name == "foo"));
    }

    #[test]
    fn test_parse_fn_with_params_and_return() {
        let prog = ok("fn add(x: f32, y: f32) -> f32 { x + y }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert_eq!(f.params.len(), 2);
        assert!(f.ret_ty.is_some());
        assert_eq!(f.params[0].name, "x");
        assert_eq!(f.params[1].name, "y");
    }

    #[test]
    fn test_parse_async_fn() {
        let prog = ok("async fn fetch() {}");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(f.is_async);
    }

    #[test]
    fn test_parse_fn_generic() {
        let prog = ok("fn id<T>(x: T) -> T { x }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert_eq!(f.generics.len(), 1);
        assert_eq!(f.generics[0].name, "T");
    }

    // ── §5  Structs / components ──────────────────────────────────────────────

    #[test]
    fn test_parse_struct() {
        let prog = ok("struct Point { x: f32, y: f32 }");
        assert!(matches!(&prog.items[0], Item::Struct(s) if s.name == "Point"));
    }

    #[test]
    fn test_parse_component() {
        let prog = ok("component Position { x: f32, y: f32, z: f32 }");
        assert!(matches!(&prog.items[0], Item::Component(c) if c.name == "Position"));
    }

    #[test]
    fn test_parse_enum() {
        let prog = ok("enum Color { Red, Green, Blue }");
        let Item::Enum(e) = &prog.items[0] else {
            panic!()
        };
        assert_eq!(e.variants.len(), 3);
        assert_eq!(e.variants[0].name, "Red");
    }

    #[test]
    fn test_parse_enum_with_data() {
        let prog = ok("enum Shape { Circle(f32), Rect { w: f32, h: f32 } }");
        let Item::Enum(e) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(&e.variants[0].fields, EnumVariantFields::Tuple(_)));
        assert!(matches!(
            &e.variants[1].fields,
            EnumVariantFields::Struct(_)
        ));
    }

    // ── §6  Types ─────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_tensor_type() {
        let prog = ok("fn f(a: tensor<f32>[128, 128]) {}");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let ty = f.params[0].ty.as_ref().unwrap();
        assert!(matches!(ty, Type::Tensor { elem: ElemType::F32, shape } if shape.len() == 2));
    }

    #[test]
    fn test_parse_tensor_dynamic_dim() {
        let prog = ok("fn f(a: tensor<f32>[_, 64]) {}");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let ty = f.params[0].ty.as_ref().unwrap();
        if let Type::Tensor { shape, .. } = ty {
            assert!(matches!(shape[0], DimExpr::Dynamic));
            assert!(matches!(shape[1], DimExpr::Lit(64)));
        }
    }

    #[test]
    fn test_parse_vec_type() {
        let prog = ok("fn f(v: vec3) {}");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            f.params[0].ty.as_ref().unwrap(),
            Type::Vec {
                size: VecSize::N3,
                family: VecFamily::Float
            }
        ));
    }

    #[test]
    fn test_parse_ref_type() {
        let prog = ok("fn f(x: &mut f32) {}");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            f.params[0].ty.as_ref().unwrap(),
            Type::Ref { mutable: true, .. }
        ));
    }

    #[test]
    fn test_parse_tuple_type() {
        let prog = ok("fn f(x: (f32, f32)) {}");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(f.params[0].ty.as_ref().unwrap(), Type::Tuple(ts) if ts.len() == 2));
    }

    #[test]
    fn test_parse_fn_ptr_type() {
        let prog = ok("fn f(cb: fn(f32) -> f32) {}");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            f.params[0].ty.as_ref().unwrap(),
            Type::FnPtr { .. }
        ));
    }

    // ── §8  Patterns ──────────────────────────────────────────────────────────

    #[test]
    fn test_parse_wildcard_pattern() {
        let prog = ok("fn f() { let _ = 1; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            &f.body.as_ref().unwrap().stmts[0],
            Stmt::Let {
                pattern: Pattern::Wildcard(_),
                ..
            }
        ));
    }

    #[test]
    fn test_parse_tuple_pattern() {
        let prog = ok("fn f() { let (a, b) = (1, 2); }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(&f.body.as_ref().unwrap().stmts[0],
            Stmt::Let { pattern: Pattern::Tuple { elems, .. }, .. } if elems.len() == 2));
    }

    #[test]
    fn test_parse_or_pattern() {
        let prog = ok("fn f(x: i32) { match x { 1 | 2 => {} _ => {} } }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Match { arms, .. } = &f.body.as_ref().unwrap().stmts[0] else {
            panic!()
        };
        assert!(matches!(&arms[0].pat, Pattern::Or { .. }));
    }

    // ── §9  Expressions ───────────────────────────────────────────────────────

    #[test]
    fn test_parse_matmul_expr() {
        let prog = ok("fn f() { let c = a @ b; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::MatMul { .. }));
    }

    #[test]
    fn test_parse_hadamard_mul() {
        let prog = ok("fn f() { let c = a .* b; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::HadamardMul { .. }));
    }

    #[test]
    fn test_parse_tensor_concat() {
        let prog = ok("fn f() { let c = a ++ b; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::TensorConcat { .. }));
    }

    #[test]
    fn test_parse_grad_expr() {
        let prog = ok("fn f() { let g = grad x; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::Grad { .. }));
    }

    #[test]
    fn test_parse_pow_expr() {
        let prog = ok("fn f() { let x = 2 ** 10; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::Pow { .. }));
    }

    #[test]
    fn test_parse_field_access() {
        let prog = ok("fn f() { let x = v.x; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::Field { field, .. } if field == "x"));
    }

    #[test]
    fn test_parse_index_expr() {
        let prog = ok("fn f() { let x = a[0]; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::Index { .. }));
    }

    #[test]
    fn test_parse_call_expr() {
        let prog = ok("fn f() { foo(1, 2, 3); }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            &f.body.as_ref().unwrap().stmts[0],
            Stmt::Expr {
                expr: Expr::Call { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_parse_method_call() {
        let prog = ok("fn f() { v.dot(w); }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            &f.body.as_ref().unwrap().stmts[0],
            Stmt::Expr {
                expr: Expr::MethodCall { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_parse_closure() {
        let prog = ok("fn f() { let g = |x| x + 1; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::Closure { .. }));
    }

    #[test]
    fn test_parse_cast_expr() {
        let prog = ok("fn f() { let x = 1 as f32; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::Cast { .. }));
    }

    #[test]
    fn test_parse_range_expr() {
        let prog = ok("fn f() { for i in 0..10 {} }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            &f.body.as_ref().unwrap().stmts[0],
            Stmt::ForIn { .. }
        ));
    }

    #[test]
    fn test_parse_if_else_expr() {
        let prog = ok("fn f() -> i32 { if true { 1 } else { 2 } }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(f.body.as_ref().unwrap().tail.is_some());
    }

    #[test]
    fn test_parse_struct_literal() {
        let prog = ok("fn f() { let p = Point { x: 1.0, y: 2.0 }; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::StructLit { name, .. } if name == "Point"));
    }

    #[test]
    fn test_parse_array_literal() {
        let prog = ok("fn f() { let a = [1, 2, 3]; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::ArrayLit { elems, .. } if elems.len() == 3));
    }

    #[test]
    fn test_parse_tuple_expr() {
        let prog = ok("fn f() { let t = (1, 2.0); }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        assert!(matches!(expr, Expr::Tuple { elems, .. } if elems.len() == 2));
    }

    #[test]
    fn test_parse_precedence_mul_before_add() {
        // 1 + 2 * 3 should parse as 1 + (2 * 3)
        let prog = ok("fn f() { let x = 1 + 2 * 3; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        if let Expr::BinOp {
            op: BinOpKind::Add,
            rhs,
            ..
        } = expr
        {
            assert!(matches!(
                rhs.as_ref(),
                Expr::BinOp {
                    op: BinOpKind::Mul,
                    ..
                }
            ));
        } else {
            panic!("expected Add at top level")
        }
    }

    #[test]
    fn test_parse_matmul_higher_than_add() {
        // a + b @ c  → a + (b @ c)
        let prog = ok("fn f() { let x = a + b @ c; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::Let {
            init: Some(expr), ..
        } = &f.body.as_ref().unwrap().stmts[0]
        else {
            panic!()
        };
        if let Expr::BinOp {
            op: BinOpKind::Add,
            rhs,
            ..
        } = expr
        {
            assert!(matches!(rhs.as_ref(), Expr::MatMul { .. }));
        } else {
            panic!("expected Add at top level, got {expr:?}")
        }
    }

    // ── §7  Statements ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_let_mut() {
        let prog = ok("fn f() { let mut x = 0; }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            &f.body.as_ref().unwrap().stmts[0],
            Stmt::Let { mutable: true, .. }
        ));
    }

    #[test]
    fn test_parse_while() {
        let prog = ok("fn f() { while x > 0 { x -= 1; } }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            &f.body.as_ref().unwrap().stmts[0],
            Stmt::While { .. }
        ));
    }

    #[test]
    fn test_parse_loop_break() {
        let prog = ok("fn f() { loop { break; } }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            &f.body.as_ref().unwrap().stmts[0],
            Stmt::Loop { .. }
        ));
    }

    #[test]
    fn test_parse_match_stmt() {
        let prog = ok("fn f(x: i32) { match x { 1 => 1, _ => 0 } }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            &f.body.as_ref().unwrap().stmts[0],
            Stmt::Match { .. }
        ));
    }

    #[test]
    fn test_parse_if_else_chain() {
        let prog = ok("fn f(x: i32) { if x > 0 { } else if x < 0 { } else { } }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let Stmt::If { else_, .. } = &f.body.as_ref().unwrap().stmts[0] else {
            panic!()
        };
        assert!(matches!(else_.as_ref().unwrap().as_ref(), IfOrBlock::If(_)));
    }

    #[test]
    fn test_parse_entity_for() {
        let prog =
            ok("system Update(dt: f32) { for entity in world with (Position, Velocity) {} }");
        let Item::System(s) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(&s.body.stmts[0], Stmt::EntityFor { query, .. }
            if query.with.len() == 2));
    }

    #[test]
    fn test_parse_entity_for_without() {
        let prog = ok("system S() { for e in world with (Health) without (Dead) {} }");
        let Item::System(s) = &prog.items[0] else {
            panic!()
        };
        let Stmt::EntityFor { query, .. } = &s.body.stmts[0] else {
            panic!()
        };
        assert_eq!(query.with, vec!["Health".to_string()]);
        assert_eq!(query.without, vec!["Dead".to_string()]);
    }

    #[test]
    fn test_parse_parallel_for() {
        let prog = ok("fn f() { parallel for x in items {} }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            &f.body.as_ref().unwrap().stmts[0],
            Stmt::ParallelFor(_)
        ));
    }

    #[test]
    fn test_parse_spawn_sync_atomic() {
        let prog = ok("fn f() { spawn { } sync { } atomic { } }");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        let stmts = &f.body.as_ref().unwrap().stmts;
        assert!(matches!(&stmts[0], Stmt::Spawn(_)));
        assert!(matches!(&stmts[1], Stmt::Sync(_)));
        assert!(matches!(&stmts[2], Stmt::Atomic(_)));
    }

    // ── §10  Agent ────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_agent_basic() {
        let prog = ok(r#"
            agent Warden {
                perception vision 40
                memory episodic 120
                learning reinforcement
                behavior Flee(priority: 100) {
                    return seek(exit)
                }
                goal Survive { self.health / 100.0 }
            }
        "#);
        let Item::Agent(a) = &prog.items[0] else {
            panic!()
        };
        assert_eq!(a.name, "Warden");
        assert_eq!(a.perceptions.len(), 1);
        assert!(a.memory.is_some());
        assert!(a.learning.is_some());
        assert_eq!(a.behaviors.len(), 1);
        assert_eq!(a.goals.len(), 1);
    }

    #[test]
    fn test_parse_agent_behavior_priority() {
        let prog = ok(r#"
            agent Guard {
                behavior Patrol(priority: 10) { }
                behavior Attack(priority: 90) { }
            }
        "#);
        let Item::Agent(a) = &prog.items[0] else {
            panic!()
        };
        assert_eq!(a.behaviors[0].priority.0, 10);
        assert_eq!(a.behaviors[1].priority.0, 90);
    }

    // ── §11  Model ────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_model_basic() {
        let prog = ok(r#"
            model PolicyNet {
                input 128
                dense 256 relu
                dense 256 relu
                output 12 softmax
            }
        "#);
        let Item::Model(m) = &prog.items[0] else {
            panic!()
        };
        assert_eq!(m.name, "PolicyNet");
        assert_eq!(m.layers.len(), 4);
        assert!(matches!(&m.layers[0], ModelLayer::Input { size: 128, .. }));
        assert!(matches!(&m.layers[1], ModelLayer::Dense { units: 256, .. }));
        assert!(matches!(&m.layers[3], ModelLayer::Output { units: 12, .. }));
    }

    #[test]
    fn test_parse_model_attention() {
        let prog = ok("model Attn { input 512 attention 8 64 output 512 linear }");
        let Item::Model(m) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(
            &m.layers[1],
            ModelLayer::Attention {
                num_heads: 8,
                head_dim: 64,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_model_dropout_norm() {
        let prog = ok("model D { input 64 dropout 0.1 norm layer output 10 softmax }");
        let Item::Model(m) = &prog.items[0] else {
            panic!()
        };
        assert!(matches!(&m.layers[1], ModelLayer::Dropout { .. }));
        assert!(matches!(
            &m.layers[2],
            ModelLayer::Norm {
                kind: NormKind::Layer,
                ..
            }
        ));
    }

    // ── §12  Train ────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_train_basic() {
        let prog = ok(r#"
            train Warden in HollowFacility {
                reward survive 1.0
                penalty seen   2.0
                episode { max_steps: 2000, num_envs: 64 }
            }
        "#);
        let Item::Train(t) = &prog.items[0] else {
            panic!()
        };
        assert_eq!(t.agent, "Warden");
        assert_eq!(t.world, "HollowFacility");
        assert_eq!(t.signals.len(), 2);
        assert!(t.signals[0].is_reward);
        assert!(!t.signals[1].is_reward);
        assert!(t.episode.is_some());
        assert_eq!(t.episode.as_ref().unwrap().max_steps, Some(2000));
        assert_eq!(t.episode.as_ref().unwrap().num_envs, Some(64));
    }

    // ── Attributes ────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_gpu_attr() {
        let prog = ok("@gpu fn forward() {}");
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(f.attrs.contains(&Attribute::Gpu));
    }

    #[test]
    fn test_parse_grad_attr() {
        let prog = ok("@grad model M { input 1 output 1 linear }");
        let Item::Model(m) = &prog.items[0] else {
            panic!()
        };
        assert!(m.attrs.contains(&Attribute::Grad));
    }

    #[test]
    fn test_parse_custom_ai_attr_with_arch_and_lr() {
        let prog = ok("@PPO(124->248->68, lr = 0.0003) agent Bot {}");
        let Item::Agent(a) = &prog.items[0] else {
            panic!()
        };
        let attr = a
            .attrs
            .iter()
            .find(|x| matches!(x, Attribute::Named { name, .. } if name == "ppo"))
            .expect("expected @ppo attribute");
        let Attribute::Named { args, .. } = attr else {
            panic!()
        };
        assert!(matches!(&args[0], Expr::StrLit { value, .. } if value == "124->248->68"));
        assert!(matches!(&args[1], Expr::Assign { .. }));
    }

    // ── Recovery from syntax errors ───────────────────────────────────────────

    #[test]
    fn test_error_recovery_continues_parsing() {
        // Missing `{` for fn body — parser should recover and still see second fn.
        let (prog, errs) = parse("fn bad( } fn good() {}");
        assert!(!errs.is_empty(), "expected at least one parse error");
        // Parser should have recovered enough to find 'good'.
        let names: Vec<_> = prog
            .items
            .iter()
            .filter_map(|i| {
                if let Item::Fn(f) = i {
                    Some(f.name.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            names.contains(&"good"),
            "expected 'good' fn after recovery: {names:?}"
        );
    }

    // ── End-to-end GPU forward pass ───────────────────────────────────────────

    #[test]
    fn test_parse_gpu_forward_snippet() {
        let src = r#"
            @gpu
            fn forward(
                A: tensor<f32>[128, 128],
                B: tensor<f32>[128, 128],
            ) -> tensor<f32>[128, 128] {
                let C = grad A @ B
                return C
            }
        "#;
        let prog = ok(src);
        let Item::Fn(f) = &prog.items[0] else {
            panic!()
        };
        assert!(f.attrs.contains(&Attribute::Gpu));
        assert_eq!(f.params.len(), 2);
        // Body: let + return
        let body = f.body.as_ref().unwrap();
        assert_eq!(body.stmts.len(), 2);
        // first stmt: let C = grad A @ B
        let Stmt::Let {
            init: Some(expr), ..
        } = &body.stmts[0]
        else {
            panic!()
        };
        // grad (A @ B)  — grad binds tighter than @? No, @ binds tighter.
        // So: (grad A) @ B  — Grad wraps ident A, then MatMul wraps that.
        assert!(matches!(expr, Expr::MatMul { lhs, .. }
            if matches!(lhs.as_ref(), Expr::Grad { .. })));
    }

    #[test]
    fn test_parse_ecs_system_snippet() {
        let src = r#"
            @simd
            system PhysicsUpdate(dt: f32) {
                for entity in world with (Position, Velocity) without (Dead) {
                    entity.position += entity.velocity * dt
                }
            }
        "#;
        let prog = ok(src);
        let Item::System(s) = &prog.items[0] else {
            panic!()
        };
        assert_eq!(s.name, "PhysicsUpdate");
        assert!(s.attrs.contains(&Attribute::Simd));
        let Stmt::EntityFor { query, .. } = &s.body.stmts[0] else {
            panic!()
        };
        assert_eq!(query.with.len(), 2);
        assert_eq!(query.without.len(), 1);
    }

    #[test]
    fn test_parse_full_program() {
        let src = r#"
            component Position { x: f32, y: f32, z: f32 }
            component Velocity { dx: f32, dy: f32, dz: f32 }
            component Health   { value: f32 }
            component Dead     {}

            @simd
            system Move(dt: f32) {
                for e in world with (Position, Velocity) without (Dead) {
                    e.position.x += e.velocity.dx * dt
                }
            }

            @gpu
            @grad
            model PolicyNet {
                input 64
                dense 128 relu
                dense 64  relu
                output 4  softmax
            }

            agent Soldier {
                perception vision 30
                learning reinforcement, model: PolicyNet
                behavior Flee(priority: 100) {
                    if self.health < 20.0 {
                        return seek(spawn_point)
                    }
                }
            }

            train Soldier in BattleField {
                reward survive 1.0
                penalty damage  0.5
                episode { max_steps: 500, num_envs: 32 }
            }
        "#;
        let prog = ok(src);
        assert_eq!(prog.items.len(), 7, "expected 7 top-level items: {prog:#?}");
    }
}
