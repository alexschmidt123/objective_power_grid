"""Adapt ONLY the native actor-update expression, preserving vendor source."""
import ast,inspect,textwrap

def build_native_class(base):
    tree=ast.parse(textwrap.dedent(inspect.getsource(base.train)))
    replaced=[]
    class ActorUpdate(ast.NodeTransformer):
        def visit_If(self,node):
            self.generic_visit(node)
            if len(node.body)!=4:return node
            rendered=[ast.unparse(v) for v in node.body]
            if rendered[1:]!=['designs = self.actor_net(X_actor)',
                              'X = torch.cat([X_actor, designs], dim=-1)',
                              'output = -self.critic_net(X).mean()']:return node
            assert ast.unparse(node.test)=='not self.use_encoder'
            node.body=[node.body[0],ast.parse('output = self.actor_net.native_actor_loss(self.critic_net, X_actor)').body[0]]
            replaced.append(rendered)
            return node
    tree=ActorUpdate().visit(tree);ast.fix_missing_locations(tree)
    if len(replaced)!=1:raise RuntimeError('Native actor block changed; refuse ambiguous adaptation')
    namespace=dict(base.train.__globals__)
    exec(compile(tree,'<V3 native categorical actor expectation>','exec'),namespace)
    adapted=type('DecisionPGvsOED',(base,),{'train':namespace['train']})
    return adapted,dict(replacements=1,old=replaced[0],new='output = self.actor_net.native_actor_loss(self.critic_net, X_actor)',
                        invariant='All other native train AST nodes are unchanged; no vendor file edits')
