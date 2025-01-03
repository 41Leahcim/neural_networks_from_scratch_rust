echo "Without rayon" && \
cargo test --release && \
cargo run --release && \
echo "With rayon" && \
cargo test --release --features rayon && \
cargo run --release --features rayon